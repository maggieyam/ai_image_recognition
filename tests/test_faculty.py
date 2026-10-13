import pytest

from brain.faculty import Faculty, check_wiring, run
from brain.workspace import Workspace


class Hear(Faculty):
    needs = ("sound",)
    gives = ("words",)

    def run(self, workspace):
        workspace.post(words=workspace["sound"].upper())  # nothing for the page


class Echo(Faculty):
    needs = ("words",)

    def run(self, workspace):
        yield {"echo": workspace["words"]}


class Look(Faculty):
    needs = ("image",)

    def run(self, workspace):
        yield {"saw": workspace["image"]}


def test_faculties_run_in_turn_and_share_only_through_the_workspace():
    workspace = Workspace(sound="hello")
    assert list(run([Hear(), Echo()], workspace)) == [{"echo": "HELLO"}]
    assert workspace["words"] == "HELLO" and "image" not in workspace


def test_a_faculty_is_skipped_when_its_sense_took_in_nothing():
    # A picture and no sound: sight still runs, and so does nothing that needs hearing.
    assert list(run([Hear(), Echo(), Look()], Workspace(image="a cat"))) == [{"saw": "a cat"}]


def test_wiring_is_checked_against_the_stimulus_and_the_faculties_before():
    check_wiring([Hear(), Echo(), Look()], ("sound", "image"))
    with pytest.raises(ValueError, match="Echo needs words"):
        check_wiring([Echo(), Hear()], ("sound",))  # in the wrong order
    with pytest.raises(ValueError, match="Look needs image"):
        check_wiring([Look()], ("sound",))  # no sense gives it
