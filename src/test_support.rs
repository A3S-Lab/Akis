#![cfg(test)]
use std::cell::Cell;

use crate::CancellationToken;

thread_local! {
    static ARMED: Cell<Option<&'static str>> = const { Cell::new(None) };
}

pub(crate) struct CancelArm;

impl Drop for CancelArm {
    fn drop(&mut self) {
        ARMED.with(|slot| slot.set(None));
    }
}

pub(crate) fn arm(stage: &'static str) -> CancelArm {
    ARMED.with(|slot| slot.set(Some(stage)));
    CancelArm
}

pub(crate) fn hit(stage: &'static str, cancel: &CancellationToken) {
    let matched = ARMED.with(|slot| slot.get() == Some(stage));
    if matched {
        ARMED.with(|slot| slot.set(None));
        cancel.cancel();
    }
}

#[path = "synthetic_fixture.rs"]
mod synthetic_fixture;

pub(crate) use synthetic_fixture::{fixture, tokenizer_blob};
