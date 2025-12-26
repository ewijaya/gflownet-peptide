"""
Loss Functions for GFlowNet Training.

Implements Trajectory Balance (TB) and Sub-Trajectory Balance (SubTB) losses.

References:
    - Bengio et al. (2021): Flow Network based Generative Models
    - Malkin et al. (2022): Trajectory Balance: Improved Credit Assignment in GFlowNets
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class TrajectoryBalanceLoss(nn.Module):
    """
    Trajectory Balance (TB) Loss for GFlowNet.

    The TB loss enforces the flow matching condition over complete trajectories:

    L_TB = (log Z + sum(log P_F) - log R - sum(log P_B))^2

    where:
    - Z is the partition function (learned)
    - P_F is the forward policy
    - R is the reward
    - P_B is the backward policy

    For uniform P_B (linear autoregressive generation), sum(log P_B) = 0,
    simplifying to: L_TB = (log Z + sum(log P_F) - log R)^2
    """

    def __init__(self, init_log_z: float = 0.0, epsilon: float = 1e-8):
        """
        Args:
            init_log_z: Initial value for log partition function
            epsilon: Small constant for numerical stability (unused, rewards
                     should already be in log space)
        """
        super().__init__()

        # Learnable log partition function
        self.log_z = nn.Parameter(torch.tensor(init_log_z))
        self.epsilon = epsilon

    def forward(
        self,
        log_pf_sum: torch.Tensor,
        log_pb_sum: torch.Tensor,
        log_rewards: torch.Tensor,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute TB loss for a batch of trajectories.

        Args:
            log_pf_sum: Sum of log forward probs [batch]
            log_pb_sum: Sum of log backward probs [batch]
            log_rewards: Log rewards for terminal states [batch]
            return_info: If True, also return info dict with metrics

        Returns:
            loss: Scalar TB loss
            info: (optional) Dictionary with detailed metrics
        """
        # TB condition: log Z + sum(log P_F) = log R + sum(log P_B)
        # Loss: (log Z + sum(log P_F) - log R - sum(log P_B))^2

        residual = self.log_z + log_pf_sum - log_rewards - log_pb_sum
        loss = (residual ** 2).mean()

        if return_info:
            with torch.no_grad():
                info = {
                    'loss': loss.item(),
                    'log_z': self.log_z.item(),
                    'mean_log_pf': log_pf_sum.mean().item(),
                    'mean_log_pb': log_pb_sum.mean().item(),
                    'mean_log_reward': log_rewards.mean().item(),
                    'mean_reward': torch.exp(log_rewards).mean().item(),
                    'max_reward': torch.exp(log_rewards).max().item(),
                    'residual_mean': residual.mean().item(),
                    'residual_std': residual.std().item(),
                }
            return loss, info

        return loss

    def get_log_z(self) -> float:
        """Return current estimate of log partition function."""
        return self.log_z.item()


class SubTrajectoryBalanceLoss(nn.Module):
    """
    Sub-Trajectory Balance (SubTB) Loss for GFlowNet.

    SubTB computes the loss on sub-trajectories rather than complete trajectories,
    which can improve credit assignment for long sequences.

    For each intermediate state, it computes:
    L_SubTB = (log F(s) + sum(log P_F from s to x) - log R - sum(log P_B from x to s))^2

    where F(s) is the estimated flow at state s.
    """

    def __init__(
        self,
        init_log_z: float = 0.0,
        lambda_sub: float = 0.9,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            init_log_z: Initial value for log partition function
            lambda_sub: Weight decay for sub-trajectory contributions
            epsilon: Small constant for numerical stability
        """
        super().__init__()

        self.log_z = nn.Parameter(torch.tensor(init_log_z))
        self.lambda_sub = lambda_sub
        self.epsilon = epsilon
        # State flow estimator for intermediate states (optional)
        self.state_flow = None

    def forward(
        self,
        log_pf_per_step: torch.Tensor,
        log_pb_per_step: torch.Tensor,
        log_rewards: torch.Tensor,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute SubTB loss.

        Args:
            log_pf_per_step: Log forward probs at each step [batch, seq_len]
            log_pb_per_step: Log backward probs at each step [batch, seq_len]
            log_rewards: Log rewards for terminal states [batch]
            return_info: If True, also return info dict with metrics

        Returns:
            loss: Scalar SubTB loss
            info: (optional) Dictionary with detailed metrics
        """
        batch_size, seq_len = log_pf_per_step.shape

        total_loss = torch.zeros(1, device=log_pf_per_step.device)

        # Compute loss for each sub-trajectory starting point
        for t in range(seq_len):
            # Cumulative log probs from step t to end
            log_pf_from_t = log_pf_per_step[:, t:].sum(dim=1)
            log_pb_from_t = log_pb_per_step[:, t:].sum(dim=1)

            # Sub-trajectory loss with log Z at step t (approximated)
            # For simplicity, we use a single log_z and decay
            weight = self.lambda_sub ** (seq_len - t - 1)
            residual = self.log_z + log_pf_from_t - log_rewards - log_pb_from_t
            total_loss = total_loss + weight * (residual ** 2).mean()

        loss = total_loss / seq_len

        if return_info:
            with torch.no_grad():
                # Compute full trajectory residual for reporting
                log_pf_sum = log_pf_per_step.sum(dim=1)
                log_pb_sum = log_pb_per_step.sum(dim=1)
                full_residual = self.log_z + log_pf_sum - log_rewards - log_pb_sum

                info = {
                    'loss': loss.item(),
                    'log_z': self.log_z.item(),
                    'mean_log_pf': log_pf_sum.mean().item(),
                    'mean_log_pb': log_pb_sum.mean().item(),
                    'mean_log_reward': log_rewards.mean().item(),
                    'mean_reward': torch.exp(log_rewards).mean().item(),
                    'residual_mean': full_residual.mean().item(),
                    'residual_std': full_residual.std().item(),
                }
            return loss, info

        return loss

    def get_log_z(self) -> float:
        """Return current estimate of log partition function."""
        return self.log_z.item()


class DetailedBalanceLoss(nn.Module):
    """
    Detailed Balance (DB) Loss for GFlowNet.

    DB enforces flow matching at each transition:
    F(s) * P_F(s'|s) = F(s') * P_B(s|s')

    This requires estimating flows at each state, which needs a separate
    flow estimator network.
    """

    def __init__(self, flow_estimator: nn.Module):
        """
        Args:
            flow_estimator: Network that estimates log F(s) for each state
        """
        super().__init__()
        self.flow_estimator = flow_estimator

    def forward(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        log_pf: torch.Tensor,
        log_pb: torch.Tensor,
        log_rewards: torch.Tensor,
        is_terminal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DB loss for transitions.

        Args:
            states: Current states [batch, state_dim]
            next_states: Next states [batch, state_dim]
            log_pf: Log forward prob [batch]
            log_pb: Log backward prob [batch]
            log_rewards: Log rewards (for terminal states) [batch]
            is_terminal: Whether next state is terminal [batch]

        Returns:
            loss: Scalar DB loss
        """
        # Estimate flows
        log_f_s = self.flow_estimator(states)
        log_f_s_prime = self.flow_estimator(next_states)

        # For terminal states, F(x) = R(x)
        log_f_s_prime = torch.where(is_terminal, log_rewards, log_f_s_prime)

        # DB condition: log F(s) + log P_F(s'|s) = log F(s') + log P_B(s|s')
        residual = log_f_s + log_pf - log_f_s_prime - log_pb
        loss = (residual ** 2).mean()

        return loss
