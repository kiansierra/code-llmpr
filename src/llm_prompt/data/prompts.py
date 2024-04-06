__all__ = ["REWRITE_PROMPTS", "REWRITE_TEMPLATES"]

# flake8: noqa

AUTHORS = [
    "Dr. Seuss",
    "William Shakespeare",
    "Tupac Shakur",
    "J.K Rowling",
    "Stephen King",
    "JRR Tolkien",
    "Paulo Coelho",
    "Taylor Swift",
    "Shakespeare",
    "Jane Austen",
    "Charles Dickens",
    "Mark Twain",
]

STYLES = ["sea shanty", "rap song", "poem", "haiku", "limerick", "sonnet", "ballad", "ode", "epic"]

HISTORICAL_PERIODS = [
    "Victorian era",
    "Elizabethan era",
    "Renaissance",
    "Medieval",
    "Ancient Greek",
    "Ancient Roman",
    "Middle Ages",
    "Enlightenment",
    "Baroque",
    "Romantic",
    "Modernist",
    "Postmodernist",
]

CHARACTERS = [
    "magician",
    "wizard",
    "necromancer",
    "witch",
    "warrior",
    "knight",
    "sorcerer",
    "elf",
    "dwarf",
    "dragon",
    "king",
    "priest",
    "queen",
    "prince",
    "princess",
    "vampire",
    "werewolf",
    "zombie",
    "ghost",
    "demon",
    "angel",
    "fairy",
    "mermaid",
    "pirate",
    "ninja",
    "samurai",
    "cowboy",
    "detective",
    "spy",
    "superhero",
    "villain",
    "monster",
    "robot",
    "alien",
    "time traveler",
    "space explorer",
    "adventurer",
    "explorer",
    "scientist",
    "inventor",
    "artist",
    "musician",
    "poet",
    "writer",
    "philosopher",
    "scholar",
    "historian",
    "journalist",
    "reporter",
    "blogger",
    "critic",
    "activist",
    "politician",
    "leader",
    "rebel",
    "revolutionary",
    "anarchist",
    "capitalist",
]

AUTHOR_PROMPTS = [
    "Rewrite this essay but do it using the writing style of {author}.",
    "Transform this text as if it was written by {author}.",
    "Imagine {author} was to rewrite this text, what would it be like.",
]

STYLE_PROMPTS = [
    "Rewrite this text in the style of a {style}.",
    "Transform this in to a {style}.",
    "How would you rewrite this text in the style of a {style}.",
]

PERIOD_PROMPTS = [
    "Rewrite the text as if it were written during the {period} historical period.",
    "Transform this text using language and cultural references appropriate to the {period} time period",
]


IMPROVE_PROMPTS = [
    "Improve this text by adding a {character} twist.",
    "Improve this text by adding a {character}.",
    "Improve this text by rewriting it from the perspective of a {character}.",
]

BASIC_REWRITE_PROMPTS = [
    "Improve the text.",
    "Make this text better.",
    "Rewrite this essay.",
    "Improve this essay.",
    "Make this essay better.",
    "Summarize this text.",
    "Rewrite the text using formal language suitable for an academic journal.",
    "Create a compelling plot that engages the reader.",
    "Rewrite the text in a humorous tone, using wit, sarcasm, and comedic elements.",
    "Use rhetorical devices and persuasive language techniques.",
    "Rewrite the text as if you're having a friendly conversation with a reader.",
    "Rewrite the text using poetic language, imagery, and metaphors.",
    "Focus on creating a rhythmic flow and evoking emotions through language.",
    "Recast the text using poetic language, metaphors, and imagery to create a rhythmic and evocative piece of writing.",
    "Rewrite the text as a travelogue, describing experiences, sights, and sounds in vivid detail to transport the reader to different locations.",
]

ALL_AUTHOR_PROMPTS = [prompt.format(author=author) for author in AUTHORS for prompt in AUTHOR_PROMPTS]
ALL_STYLE_PROMPTS = [prompt.format(style=style) for style in STYLES for prompt in STYLE_PROMPTS]
ALL_PERIOD_PROMPTS = [prompt.format(period=period) for period in HISTORICAL_PERIODS for prompt in PERIOD_PROMPTS]
ALL_IMPROVE_PROMPTS = [prompt.format(character=character) for character in CHARACTERS for prompt in IMPROVE_PROMPTS]


REWRITE_PROMPTS = {
    "author": ALL_AUTHOR_PROMPTS,
    "style": ALL_STYLE_PROMPTS,
    "period": ALL_PERIOD_PROMPTS,
    "basic": BASIC_REWRITE_PROMPTS,
    "impove": ALL_IMPROVE_PROMPTS,
}

REWRITE_TEMPLATES = [
    "{rewrite_prompt} {original_text}",
    "Given the prompt: {rewrite_prompt}, rewrite the following text: {original_text}",
    "Given the prompt: {rewrite_prompt}, rewrite the following text, don't mention anything about the task at hand: {original_text}",
]
