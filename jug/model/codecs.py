"""
I/O Codecs for Parameter Values
===============================

Codecs handle conversion between .par file representation and internal storage.

Key principle: ALL angles are stored internally as RADIANS.
Codecs operate ONLY at I/O boundary (reading/writing .par files, GUI display).

Usage:
    from jug.model.codecs import CODECS

    # Decode from .par file
    raj_rad = CODECS['raj'].decode("19:09:47.4346970")

    # Encode for .par file output
    raj_str = CODECS['raj'].encode(5.01234567)

Codec types:
- FloatCodec: Simple float conversion
- EpochMJDCodec: MJD epoch handling
- RAJCodec: Right ascension (HH:MM:SS.sss <-> radians)
- DECJCodec: Declination (DD:MM:SS.sss <-> radians)
"""

from abc import ABC, abstractmethod
import math
import re
from typing import Union


class Codec(ABC):
    """
    Abstract base class for parameter codecs.

    A codec converts between .par file representation (string)
    and internal representation (float).
    """

    @abstractmethod
    def decode(self, par_value: str) -> float:
        """
        Decode a .par file value to internal representation.

        Parameters
        ----------
        par_value : str
            Value as it appears in .par file

        Returns
        -------
        float
            Internal representation
        """
        pass

    @abstractmethod
    def encode(self, internal_value: float) -> str:
        """
        Encode internal value to .par file representation.

        Parameters
        ----------
        internal_value : float
            Internal representation

        Returns
        -------
        str
            Value for .par file
        """
        pass


class FloatCodec(Codec):
    """
    Simple float codec - no transformation.

    Used for parameters where .par file value equals internal value.
    """

    def decode(self, par_value: str) -> float:
        """Parse string as float."""
        return float(par_value)

    def encode(self, internal_value: float) -> str:
        """Format float as string with full precision."""
        # Use repr for full precision
        if internal_value == 0.0:
            return "0.0"
        # Scientific notation for very small/large values
        if abs(internal_value) < 1e-10 or abs(internal_value) > 1e10:
            return f"{internal_value:.20e}"
        return f"{internal_value:.20f}".rstrip('0').rstrip('.')


class EpochMJDCodec(Codec):
    """
    Epoch MJD codec - preserves precision.

    MJD epochs are stored as-is (no unit conversion).
    """

    def decode(self, par_value: str) -> float:
        """Parse MJD epoch."""
        return float(par_value)

    def encode(self, internal_value: float) -> str:
        """Format MJD with appropriate precision."""
        # MJD typically needs ~5-6 decimal places for sub-second precision
        return f"{internal_value:.10f}".rstrip('0').rstrip('.')


class RAJCodec(Codec):
    """
    Right Ascension codec: HH:MM:SS.sss <-> radians

    .par format: "HH:MM:SS.ssssssss" (sexagesimal)
    Internal: radians [0, 2*pi)

    Examples:
        "19:09:47.4346970" -> 5.0116... radians
        5.0116 radians -> "19:09:47.4346970"
    """

    # Regex for parsing RA
    _PATTERN = re.compile(r'^(\d+):(\d+):(\d+(?:\.\d+)?)$')

    def decode(self, par_value: str) -> float:
        """
        Convert sexagesimal RA to radians.

        Parameters
        ----------
        par_value : str
            RA in format "HH:MM:SS.sss"

        Returns
        -------
        float
            RA in radians

        Raises
        ------
        ValueError
            If format is invalid
        """
        par_value = par_value.strip()

        # Try to parse as sexagesimal
        match = self._PATTERN.match(par_value)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = float(match.group(3))

            # Validate ranges
            if not (0 <= hours < 24):
                raise ValueError(f"Hours must be 0-23, got {hours}")
            if not (0 <= minutes < 60):
                raise ValueError(f"Minutes must be 0-59, got {minutes}")
            if not (0 <= seconds < 60):
                raise ValueError(f"Seconds must be 0-60, got {seconds}")

            # Convert to radians: RA_rad = RA_hours * (2*pi / 24)
            ra_hours = hours + minutes / 60.0 + seconds / 3600.0
            return ra_hours * (math.pi / 12.0)

        # Try as raw radians (for convenience)
        try:
            return float(par_value)
        except ValueError:
            raise ValueError(f"Cannot parse RA: '{par_value}'")

    def encode(self, internal_value: float) -> str:
        """
        Convert radians to sexagesimal RA.

        Parameters
        ----------
        internal_value : float
            RA in radians

        Returns
        -------
        str
            RA in format "HH:MM:SS.sssssss"
        """
        # Convert radians to hours: RA_hours = RA_rad * (12 / pi)
        ra_hours = internal_value * (12.0 / math.pi)

        # Normalize to [0, 24)
        ra_hours = ra_hours % 24.0

        hours = int(ra_hours)
        minutes_float = (ra_hours - hours) * 60.0
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60.0

        # Handle floating point edge cases
        if seconds >= 59.99999995:
            seconds = 0.0
            minutes += 1
        if minutes >= 60:
            minutes = 0
            hours += 1
        if hours >= 24:
            hours = 0

        return f"{hours:02d}:{minutes:02d}:{seconds:011.8f}"


class DECJCodec(Codec):
    """
    Declination codec: DD:MM:SS.sss <-> radians

    .par format: "±DD:MM:SS.ssssssss" (sexagesimal)
    Internal: radians [-pi/2, pi/2]

    Examples:
        "-37:44:14.46622" -> -0.6586... radians
        -0.6586 radians -> "-37:44:14.46622"
    """

    # Regex for parsing DEC (handles optional sign)
    _PATTERN = re.compile(r'^([+-]?)(\d+):(\d+):(\d+(?:\.\d+)?)$')

    def decode(self, par_value: str) -> float:
        """
        Convert sexagesimal DEC to radians.

        Parameters
        ----------
        par_value : str
            DEC in format "±DD:MM:SS.sss"

        Returns
        -------
        float
            DEC in radians

        Raises
        ------
        ValueError
            If format is invalid
        """
        par_value = par_value.strip()

        # Try to parse as sexagesimal
        match = self._PATTERN.match(par_value)
        if match:
            sign = -1 if match.group(1) == '-' else 1
            degrees = int(match.group(2))
            minutes = int(match.group(3))
            seconds = float(match.group(4))

            # Validate ranges
            if not (0 <= degrees <= 90):
                raise ValueError(f"Degrees must be 0-90, got {degrees}")
            if not (0 <= minutes < 60):
                raise ValueError(f"Minutes must be 0-59, got {minutes}")
            if not (0 <= seconds < 60):
                raise ValueError(f"Seconds must be 0-60, got {seconds}")

            # Convert to radians
            dec_degrees = degrees + minutes / 60.0 + seconds / 3600.0
            return sign * dec_degrees * (math.pi / 180.0)

        # Try as raw radians (for convenience)
        try:
            return float(par_value)
        except ValueError:
            raise ValueError(f"Cannot parse DEC: '{par_value}'")

    def encode(self, internal_value: float) -> str:
        """
        Convert radians to sexagesimal DEC.

        Parameters
        ----------
        internal_value : float
            DEC in radians

        Returns
        -------
        str
            DEC in format "±DD:MM:SS.sssssss"
        """
        # Convert radians to degrees
        dec_degrees = internal_value * (180.0 / math.pi)

        # Handle sign
        sign = '-' if dec_degrees < 0 else '+'
        dec_degrees = abs(dec_degrees)

        degrees = int(dec_degrees)
        minutes_float = (dec_degrees - degrees) * 60.0
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60.0

        # Handle floating point edge cases
        if seconds >= 59.99999995:
            seconds = 0.0
            minutes += 1
        if minutes >= 60:
            minutes = 0
            degrees += 1

        return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:010.7f}"


# =============================================================================
# Codec Registry
# =============================================================================

CODECS = {
    'float': FloatCodec(),
    'epoch_mjd': EpochMJDCodec(),
    'raj': RAJCodec(),
    'decj': DECJCodec(),
}


def get_codec(name: str) -> Codec:
    """
    Get a codec by name.

    Parameters
    ----------
    name : str
        Codec name (float, epoch_mjd, raj, decj)

    Returns
    -------
    Codec
        The codec instance

    Raises
    ------
    KeyError
        If codec not found
    """
    return CODECS[name]
