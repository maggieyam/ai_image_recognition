"""
The workspace: what the brain has in mind about one stimulus (a picture, for
now). Faculties never call each other. Each reads what it needs from here and
posts what it found, so one can be added or replaced without touching the
others. The idea is the "global workspace" of Baars and Dehaene: many
specialists, and one shared place where their results meet.

Contents are kept in whatever form is most faithful (the picture itself,
scores, face boxes), and as text only where text is the natural form: the
brain doesn't turn everything into words to understand it.
"""


class Workspace:
    def __init__(self, **stimulus):
        """stimulus: what the senses took in, by name (image=...)."""
        self.contents = dict(stimulus)

    def post(self, **contents):
        self.contents.update(contents)

    def __getitem__(self, name):
        return self.contents[name]

    def __contains__(self, name):
        return name in self.contents
