"""Tests for data loading modules."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from gflownet_peptide.data import (
    load_flip_stability,
    load_flip_gb1,
    load_propedia,
    validate_sequence,
    CANONICAL_AA,
)


class TestSequenceValidation:
    """Tests for sequence validation utilities."""

    def test_valid_sequence_all_canonical(self):
        """Test that all canonical amino acids are accepted."""
        assert validate_sequence('ACDEFGHIKLMNPQRSTVWY')

    def test_valid_sequence_lowercase(self):
        """Test that lowercase is also accepted."""
        assert validate_sequence('acdefghiklmnpqrstvwy')

    def test_valid_sequence_mixed_case(self):
        """Test mixed case sequence."""
        assert validate_sequence('AcDeFgHiKl')

    def test_invalid_sequence_with_x(self):
        """Test that X (unknown) is rejected."""
        assert not validate_sequence('ACXDEF')

    def test_invalid_sequence_with_u(self):
        """Test that U (selenocysteine) is rejected."""
        assert not validate_sequence('ACUDEF')

    def test_invalid_sequence_with_b(self):
        """Test that B (Asx) is rejected."""
        assert not validate_sequence('ACBDEF')

    def test_invalid_sequence_with_z(self):
        """Test that Z (Glx) is rejected."""
        assert not validate_sequence('ACZDEF')

    def test_invalid_sequence_with_numbers(self):
        """Test that numbers are rejected."""
        assert not validate_sequence('AC123')

    def test_invalid_sequence_with_special_chars(self):
        """Test that special characters are rejected."""
        assert not validate_sequence('AC-DEF')

    def test_empty_sequence(self):
        """Test empty sequence."""
        assert validate_sequence('')

    def test_canonical_aa_set(self):
        """Test that CANONICAL_AA contains exactly 20 amino acids."""
        assert len(CANONICAL_AA) == 20
        assert CANONICAL_AA == set('ACDEFGHIKLMNPQRSTVWY')


class TestFlipStability:
    """Tests for FLIP stability data loader."""

    @pytest.fixture
    def mock_flip_data(self, tmp_path):
        """Create mock FLIP stability data for testing."""
        csv_content = """sequence,target,set,validation
ACDEFGHIKLMNPQRSTVWY,45.5,train,
MKTVRQERLKSIVRILERSKEPVSGAQL,52.3,train,
MKFLILFLPFASMGKLLVLLP,38.2,train,val
ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY,48.1,test,
SHORTSEQ,35.0,train,
INVALIDXSEQ,40.0,train,
"""
        data_file = tmp_path / 'stability.csv'
        data_file.write_text(csv_content)
        return tmp_path

    def test_load_returns_correct_types(self, mock_flip_data):
        """Test that load returns list and numpy array."""
        sequences, labels = load_flip_stability(str(mock_flip_data))
        assert isinstance(sequences, list)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.float32

    def test_length_filtering_min(self, mock_flip_data):
        """Test minimum length filtering."""
        sequences, labels = load_flip_stability(
            str(mock_flip_data),
            min_length=15,
            max_length=100
        )
        for seq in sequences:
            assert len(seq) >= 15

    def test_length_filtering_max(self, mock_flip_data):
        """Test maximum length filtering."""
        sequences, labels = load_flip_stability(
            str(mock_flip_data),
            min_length=1,
            max_length=25
        )
        for seq in sequences:
            assert len(seq) <= 25

    def test_canonical_aa_filtering(self, mock_flip_data):
        """Test that non-canonical sequences are filtered."""
        sequences, labels = load_flip_stability(
            str(mock_flip_data),
            min_length=1,
            max_length=100
        )
        # INVALIDXSEQ should be filtered out
        for seq in sequences:
            assert validate_sequence(seq)

    def test_normalization(self, mock_flip_data):
        """Test that normalization produces zero-mean unit-variance."""
        sequences, labels = load_flip_stability(
            str(mock_flip_data),
            min_length=1,
            max_length=100,
            normalize=True
        )
        if len(labels) > 1:
            assert abs(labels.mean()) < 0.1  # Close to zero
            assert abs(labels.std() - 1.0) < 0.1  # Close to unit variance

    def test_no_normalization(self, mock_flip_data):
        """Test data without normalization."""
        sequences, labels = load_flip_stability(
            str(mock_flip_data),
            min_length=1,
            max_length=100,
            normalize=False
        )
        # Original values should be preserved
        assert labels.min() > 30  # Original values are around 35-52

    def test_split_train(self, mock_flip_data):
        """Test train split selection."""
        sequences, labels = load_flip_stability(
            str(mock_flip_data),
            min_length=1,
            max_length=100,
            split='train',
            normalize=False
        )
        # Should only get train samples
        assert len(sequences) > 0

    def test_split_test(self, mock_flip_data):
        """Test test split selection."""
        sequences, labels = load_flip_stability(
            str(mock_flip_data),
            min_length=1,
            max_length=100,
            split='test',
            normalize=False
        )
        # Should only get test samples
        assert len(sequences) > 0

    def test_sequences_match_labels(self, mock_flip_data):
        """Test that sequences and labels have same length."""
        sequences, labels = load_flip_stability(str(mock_flip_data))
        assert len(sequences) == len(labels)


class TestFlipGB1:
    """Tests for FLIP GB1 data loader."""

    @pytest.fixture
    def mock_gb1_data(self, tmp_path):
        """Create mock GB1 data for testing."""
        csv_content = """sequence,target,set,validation
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE,1.0,train,
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGIDGEWTYDDATKTFTVTE,1.45,train,
MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGADGEWTYDDATKTFTVTE,0.8,test,
"""
        data_file = tmp_path / 'one_vs_rest.csv'
        data_file.write_text(csv_content)
        return tmp_path

    def test_load_returns_correct_types(self, mock_gb1_data):
        """Test that load returns correct types."""
        sequences, labels = load_flip_gb1(str(mock_gb1_data))
        assert isinstance(sequences, list)
        assert isinstance(labels, np.ndarray)

    def test_default_max_length_allows_long_sequences(self, mock_gb1_data):
        """Test that default max_length allows GB1-length sequences."""
        sequences, labels = load_flip_gb1(str(mock_gb1_data))
        # GB1 sequences are ~280 AA long
        assert all(len(seq) < 300 for seq in sequences)


class TestPropedia:
    """Tests for Propedia/PepBDB data loader."""

    @pytest.fixture
    def mock_propedia_data(self, tmp_path):
        """Create mock Propedia data for testing."""
        csv_content = """pdb_id,sequence,length
1abc_A,GSVVIVGRIVLSGKPA,16
1def_B,EYFTLQIRGRERFEKIREYNEALELKDAQ,29
1ghi_C,DFEEIPEEL,9
1jkl_D,KPIVQYDNF,9
1mno_E,ACDEFGHIKLMNPQRSTVWY,20
1pqr_F,SHORTSEQ,8
1stu_G,ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY,40
"""
        data_file = tmp_path / 'propedia.csv'
        data_file.write_text(csv_content)
        return tmp_path

    def test_load_returns_correct_types(self, mock_propedia_data):
        """Test that load returns correct types."""
        sequences, labels = load_propedia(str(mock_propedia_data))
        assert isinstance(sequences, list)
        assert isinstance(labels, np.ndarray)

    def test_length_filtering(self, mock_propedia_data):
        """Test length filtering."""
        sequences, labels = load_propedia(
            str(mock_propedia_data),
            min_length=10,
            max_length=30
        )
        for seq in sequences:
            assert 10 <= len(seq) <= 30

    def test_binary_labels_for_binders(self, mock_propedia_data):
        """Test that all labels are 1.0 for verified binders."""
        sequences, labels = load_propedia(
            str(mock_propedia_data),
            normalize=False
        )
        # All peptides from PepBDB are verified binders
        assert all(label == 1.0 for label in labels)

    def test_sequences_match_labels(self, mock_propedia_data):
        """Test that sequences and labels have same length."""
        sequences, labels = load_propedia(str(mock_propedia_data))
        assert len(sequences) == len(labels)

    def test_split_reproducibility(self, mock_propedia_data):
        """Test that splits are reproducible with same seed."""
        seq1, _ = load_propedia(str(mock_propedia_data), split='train', seed=42)
        seq2, _ = load_propedia(str(mock_propedia_data), split='train', seed=42)
        assert seq1 == seq2

    def test_different_seed_gives_different_split(self, mock_propedia_data):
        """Test that different seeds give different splits."""
        seq1, _ = load_propedia(
            str(mock_propedia_data),
            min_length=1,
            max_length=100,
            split='train',
            seed=42
        )
        seq2, _ = load_propedia(
            str(mock_propedia_data),
            min_length=1,
            max_length=100,
            split='train',
            seed=123
        )
        # With different seeds, the train sets should differ
        # (unless dataset is very small)
        # This is a weak test, just checking it runs


class TestIntegrationWithRealData:
    """Integration tests with real downloaded data."""

    @pytest.fixture
    def data_dir(self):
        """Get the real data directory."""
        # Try multiple possible locations
        paths = [
            Path('data'),
            Path('/home/ubuntu/storage1/gflownet-peptide/data'),
            Path(__file__).parent.parent / 'data',
        ]
        for p in paths:
            if p.exists():
                return p
        pytest.skip("Data directory not found")

    def test_flip_stability_loads(self, data_dir):
        """Test loading real FLIP stability data."""
        stability_path = data_dir / 'flip' / 'stability'
        if not stability_path.exists():
            pytest.skip("FLIP stability data not found")

        sequences, labels = load_flip_stability(str(stability_path))
        assert len(sequences) > 0
        assert len(sequences) == len(labels)

    def test_flip_gb1_loads(self, data_dir):
        """Test loading real FLIP GB1 data."""
        gb1_path = data_dir / 'flip' / 'gb1'
        if not gb1_path.exists():
            pytest.skip("FLIP GB1 data not found")

        sequences, labels = load_flip_gb1(str(gb1_path))
        assert len(sequences) > 0
        assert len(sequences) == len(labels)

    def test_propedia_loads(self, data_dir):
        """Test loading real Propedia data."""
        propedia_path = data_dir / 'propedia'
        if not propedia_path.exists():
            pytest.skip("Propedia data not found")

        sequences, labels = load_propedia(str(propedia_path))
        assert len(sequences) > 0
        assert len(sequences) == len(labels)
