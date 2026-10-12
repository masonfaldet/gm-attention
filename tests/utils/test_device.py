"""Tests for gm_attention.utils.device."""

from __future__ import annotations

import torch

from gm_attention.utils.device import resolve_device


def test_explicit_string_is_resolved() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_torch_device_is_passed_through() -> None:
    device = torch.device("cpu")
    assert resolve_device(device) == device


def test_none_auto_detects_cuda_availability() -> None:
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_device(None) == expected


def test_none_is_the_default() -> None:
    assert resolve_device() == resolve_device(None)
