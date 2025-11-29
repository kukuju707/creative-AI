class Tag:
    def __init__(self, string):
        self.string = string

    def __new__(cls, string):
        # reuse the same tag from the TagTable instance - and also raise an error if the user tries to use undefined tag.
        if not TagTable.get().is_initializing():
            if TagTable.has_tag_string(string):
                return TagTable.get_tag_for_string(string, False)
            raise KeyError(f"Tag string {string} is not defined anywhere - you can't define a new tag in runtime.")
        # bypass for internal setup
        return super().__new__(cls)

    def __hash__(self):
        return hash(self.string)

    def __eq__(self, other : 'Tag'):
        return self.string == other.string

class TagContainer:

    def __init__(self):
        self.tags : set[Tag] = set()

    def __eq__(self, other):
        return self.tags == other.tags

    def add_tag(self, tag : Tag):
        self.tags.add(tag)

    def remove_tag(self, tag : Tag):
        self.tags.remove(tag)

    def get_tags(self):
        return self.tags

    def has_tag(self, tag : Tag):
        return tag in self.tags

    def has_any_tag(self, tags : 'TagContainer'):
        # check if any of the tags in the in container overlaps with self tags.
        for tag in tags:
            if tag in self.tags:
                return True

        return False

    def has_all_tag(self, tags : 'TagContainer'):
        # basically the same tag containers means they overlap all.
        return self == tags

# a class that contains all the predefined tags - only the tags that are in this containers can be used.
class TagTable:
    _instance = None
    _initializing = False

    def __new__(cls):
        if cls._instance is None:
            cls._initializing = True
            cls._instance = super(TagTable, cls).__new__(cls)
            cls._instance.tags: dict[str, Tag] = TagTable._get_default_tags()
            cls._initializing = False
        return cls._instance

    def is_initializing(self):
        return self._initializing

    @classmethod
    def get(cls) -> 'TagTable':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_tag_for_string(cls, tag_string : str, b_do_check = True) -> Tag:
        if b_do_check :
            if TagTable.get().has_tag_string(tag_string):
                return TagTable.get().tags[tag_string]
            return None
        else:
            return TagTable.get().tags[tag_string]

    @classmethod
    def has_tag_string(cls, tag_string : str) -> bool:
        return tag_string in TagTable.get().tags

    @classmethod
    def has_tag(cls, tag : Tag) -> bool:
        return tag.string in TagTable.get().tags

    @classmethod
    def _get_default_tags(cls):
        tags : dict[str,Tag] = {}

        tags["LLM.TurnTaking"] = Tag("LLM.TurnTaking")
        tags["LLM.Default"] = Tag("LLM.Default")

        return tags