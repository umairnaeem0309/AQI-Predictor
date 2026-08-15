"""
Model Lifecycle — State machine for model lifecycle management.

Valid transitions:
    TRAINING → EVALUATED → REGISTERED → PRODUCTION → ARCHIVED
                                           ↓
                                        STAGING → PRODUCTION
                                           ↓
                                        REJECTED

Invalid transitions raise LifecycleTransitionError.

Synthetic data is blocked from REGISTERED and PRODUCTION states.
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """Model lifecycle states."""

    TRAINING = "training"
    EVALUATED = "evaluated"
    REGISTERED = "registered"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    REJECTED = "rejected"


# Valid transitions: from_state → list of valid to_states
VALID_TRANSITIONS = {
    ModelState.TRAINING: [ModelState.EVALUATED],
    ModelState.EVALUATED: [ModelState.REGISTERED, ModelState.REJECTED],
    ModelState.REGISTERED: [ModelState.STAGING, ModelState.PRODUCTION, ModelState.ARCHIVED],
    ModelState.STAGING: [ModelState.PRODUCTION, ModelState.ARCHIVED, ModelState.REJECTED],
    ModelState.PRODUCTION: [ModelState.ARCHIVED],
    ModelState.ARCHIVED: [],  # Terminal state
    ModelState.REJECTED: [],  # Terminal state
}

# States that require real data (not synthetic)
REAL_DATA_REQUIRED_STATES = {ModelState.REGISTERED, ModelState.PRODUCTION}


class LifecycleTransitionError(Exception):
    """Raised when an invalid lifecycle transition is attempted."""
    pass


class LifecycleBlockError(Exception):
    """Raised when synthetic data is blocked from a state."""
    pass


class ModelLifecycle:
    """State machine for model lifecycle management.

    Usage:
        lifecycle = ModelLifecycle()
        lifecycle.transition(ModelState.TRAINING, ModelState.EVALUATED)
        lifecycle.transition(ModelState.EVALUATED, ModelState.REGISTERED)
    """

    def __init__(
        self,
        model_name: str = "",
        current_state: ModelState = ModelState.TRAINING,
        dataset_type: str = "",
        is_reportable: bool = False,
    ):
        """Initialize lifecycle.

        Args:
            model_name: Name of the model.
            current_state: Initial state.
            dataset_type: Type of training dataset.
            is_reportable: Whether results can be reported.
        """
        self.model_name = model_name
        self.current_state = current_state
        self.dataset_type = dataset_type
        self.is_reportable = is_reportable
        self._history = [(current_state, "initialized")]

    def transition(
        self,
        to_state: ModelState,
        reason: str = "",
    ) -> None:
        """Execute a lifecycle transition.

        Args:
            to_state: Target state.
            reason: Reason for transition.

        Raises:
            LifecycleTransitionError: If transition is invalid.
            LifecycleBlockError: If synthetic data is blocked.
        """
        from_state = self.current_state

        # Check: synthetic data blocked from REGISTERED/PRODUCTION
        if to_state in REAL_DATA_REQUIRED_STATES:
            if self.dataset_type == "synthetic_test_data":
                raise LifecycleBlockError(
                    f"Cannot transition to {to_state.value} with synthetic test data. "
                    f"Model: {self.model_name}"
                )

        # Check: valid transition
        valid_targets = VALID_TRANSITIONS.get(from_state, [])
        if to_state not in valid_targets:
            raise LifecycleTransitionError(
                f"Invalid transition: {from_state.value} → {to_state.value}. "
                f"Valid targets from {from_state.value}: "
                f"{[s.value for s in valid_targets]}"
            )

        # Execute transition
        self.current_state = to_state
        self._history.append((to_state, reason))
        logger.info(
            "Lifecycle transition: %s → %s (model=%s, reason=%s)",
            from_state.value,
            to_state.value,
            self.model_name,
            reason,
        )

    def get_state(self) -> ModelState:
        """Get current state."""
        return self.current_state

    def get_history(self):
        """Get transition history."""
        return self._history.copy()

    def can_transition_to(self, to_state: ModelState) -> bool:
        """Check if transition to target state is valid."""
        valid_targets = VALID_TRANSITIONS.get(self.current_state, [])
        return to_state in valid_targets

    def is_terminal(self) -> bool:
        """Check if current state is terminal (no outgoing transitions)."""
        return len(VALID_TRANSITIONS.get(self.current_state, [])) == 0

    def __repr__(self) -> str:
        return (
            f"ModelLifecycle(model={self.model_name}, "
            f"state={self.current_state.value}, "
            f"dataset={self.dataset_type})"
        )
