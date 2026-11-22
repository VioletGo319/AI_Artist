from collections import Counter
from random import Random
import re

THEMES = {
    "Surrealism": ("A tide rising inside a porcelain room", "The Room Keeps the Tide", "surrealist still life"),
    "Impossible Architecture": ("A stairwell that folds through its own reflection", "The Folded Stair", "architectural study"),
    "Alien Worlds": ("A ringed moon above an estuary of rust-red reeds", "Reeds of the Outer Moon", "speculative landscape"),
    "Cosmic Art": ("An eclipse threaded with luminous tidal filaments", "A Thread Through Night", "celestial photography"),
    "Abstract Geometry": ("Nested ceramic rings crossing a plane of textured ink", "Orbital Grammar", "minimalist abstraction"),
    "Crystal Structures": ("A cathedral of translucent mineral blades in an ice cave", "Crystal Cathedral", "mineral sculpture"),
    "Futuristic Cities": ("An elevated city with radiant public gardens after rain", "Gardens Above the Rain", "architectural concept art"),
    "Dreamlike Landscapes": ("A lake floating between layers of fog and black sand", "Where the Lake Drifts", "atmospheric landscape"),
    "Macro Photography": ("Tiny droplets suspended in the branching veins of a leaf", "Vessels of Morning", "macro photography"),
    "Experimental Fine Art": ("Layered dye transfers on folded paper and polished stone", "A Study in Transfer", "mixed-media fine art"),
    "Liquid Metal": ("Molten silver stretched across cracked black glaze", "Silver at Rest", "sculptural still life"),
    "Mathematical Fractals": ("Recursive branch forms etched in translucent resin", "The Recursive Grove", "mathematical art"),
}
PALETTES = (
    "copper and midnight blue", "opal lavender and charcoal", "moss and warm ivory",
    "ink blue and amber", "ultramarine and rose quartz", "pale cyan and ember red",
    "deep plum and brushed silver", "ochre and verdigris", "saffron and dark teal",
    "oxidized silver and crimson", "lapis and ice blue", "black cherry and cream",
)
COMPOSITIONS = (
    "layered perspective with a low horizon", "centered silhouette with generous negative space",
    "asymmetrical framing and a diagonal focal line", "cropped detail expanding beyond the frame",
    "a wide view with alternating masses and voids", "an intimate view with soft foreground occlusion",
    "an overhead view with a single off-center focal point", "strong verticals and a small distant focal point",
)
DETAILS = (
    "raking light and tactile detail", "diffused daylight and muted highlights",
    "precise silhouettes with a soft atmospheric falloff", "fine grain and luminous shadows",
    "a restrained highlight bloom and sharply resolved surfaces",
)
FEATURES = ("theme", "palette", "composition", "style", "model")


def keywords(prompt):
    return sorted(set(re.findall(r"\b[a-z][a-z-]{4,}\b", prompt.lower())))


def preference_snapshot(art):
    return {**{key: art[key] for key in FEATURES}, "seed": art["seed"],
            "keywords": keywords(art["input_prompt"])}


def inspiration(seed, count, previous, approvals, exploration, exposures=()):
    rng = Random(seed)
    likes = Counter((key, value) for row in approvals for key in FEATURES
                    if (value := row.get(key)))
    seen = Counter((key, value) for row in exposures for key in FEATURES
                   if (value := row.get(key)))
    recent_keywords = Counter(word for row in approvals for word in row.get("keywords", ()))
    previous_pairs = {(slot["theme"], slot["palette"]) for slot in previous}
    combinations = [(theme, palette) for theme in THEMES for palette in PALETTES
                    if (theme, palette) not in previous_pairs]
    available = combinations[:]
    items = []
    for _ in range(count):
        # Prefer unique themes and palettes within the batch while there are enough choices.
        used_themes = {item["theme"] for item in items}
        used_colors = {item["palette"] for item in items}
        pool = [(theme, color) for theme, color in available
                if theme not in used_themes and color not in used_colors] or available
        theme, palette = choose(rng, pool, lambda item: (
            weight(likes, seen, "theme", item[0])
            * weight(likes, seen, "palette", item[1])
            * weight(likes, seen, "style", THEMES[item[0]][2])
            * (1 + min(6, sum(recent_keywords[word] for word in keywords(THEMES[item[0]][0]))) / 12)
        ), exploration)
        available.remove((theme, palette))
        subject, title, style = THEMES[theme]
        composition = choose(rng, COMPOSITIONS, lambda item: weight(likes, seen, "composition", item), exploration)
        detail = rng.choice(DETAILS)
        items.append(dict(theme=theme, palette=palette, composition=composition, style=style, title=title,
                          prompt=f"{subject}. {style.capitalize()} using {palette}, {composition}, {detail}."))
    return items


def weight(likes, seen, feature, value):
    return 1 + 3 * (likes[feature, value] + 1) / (max(likes[feature, value], seen[feature, value]) + 4)


def choose(rng, choices, get_weight, exploration):
    if rng.random() < exploration:
        return rng.choice(choices)
    return rng.choices(choices, weights=[get_weight(item) for item in choices], k=1)[0]
