"""
A faculty is one specialist of the brain: seeing, reading faces, feeling a
mood, saying something. It says what it needs from the workspace and what it
gives, and knows nothing else about the rest of the brain.
"""


class Faculty:
    needs = ()  # what it reads from the workspace; it's skipped unless all are there
    gives = ()  # what it posts to the workspace

    def run(self, workspace):
        """
        Do the work: read `needs` from the workspace and post `gives` to it.
        A faculty with something for whoever is listening (the page) yields
        it as events, plain dicts that can be sent as JSON:
        - what it found: {"caption": ..., "description": ...};
        - something it says, under its section: {"saw": "I saw a grey cat."};
        - what it's doing meanwhile, shown where a section will appear:
          {"status": "Looking at their faces…", "for": "feelings"}.
        """
        raise NotImplementedError

    def __str__(self):
        return type(self).__name__


def run(faculties, workspace):
    """
    Each faculty in turn on one workspace, yielding their events. A faculty
    is skipped when something it needs isn't there (a sense that took in
    nothing this time, say).
    """
    for faculty in faculties:
        if all(name in workspace for name in faculty.needs):
            yield from faculty.run(workspace) or ()


def check_wiring(faculties, stimulus):
    """
    Raise ValueError if a faculty needs something that neither the stimulus
    (the names the workspace can start with) nor a faculty before it gives.
    Checked when the brain starts, so a misspelt name isn't a faculty that
    silently never runs.
    """
    there = set(stimulus)
    for faculty in faculties:
        missing = sorted(set(faculty.needs) - there)
        if missing:
            raise ValueError(
                f"{faculty} needs {', '.join(missing)}, which nothing before it gives"
            )
        there.update(faculty.gives)
