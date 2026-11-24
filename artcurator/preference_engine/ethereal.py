"""A compositional brief, not a whitelist of fantasy subjects."""

PALETTES = (
    "pearl white and glacier blue with a small luminous amber accent",
    "mist lavender and deep indigo with a restrained coral glow",
    "pale jade and warm ivory with a single emerald focal accent",
    "silver grey and rose quartz with translucent turquoise light",
    "moonlit ultramarine and porcelain white with a small gold reflection",
    "soft apricot and cool slate with a luminous violet focal note",
    "celadon and distant blue with a concentrated sunlit vermilion accent",
    "opal cyan and pale lilac with warm light at the focal point",
    "smoky plum and pale silver with a small iridescent green accent",
    "warm sand and distant aquamarine with a soft rose reflection",
    "cool ink and milk white with a delicate saffron glow",
    "frosted mint and muted mauve with a vivid peach highlight",
)
COMPOSITIONS = (
    "a small off-center subject in the lower third, with two-thirds open sky or mist",
    "a low horizon, a thin foreground edge and a distant focal point across a wide quiet void",
    "one slender vertical form surrounded by generous negative space, with a faint distant echo",
    "an asymmetrical arc leading from a quiet foreground into a tiny illuminated focal point",
    "three sparse depth planes separated by atmosphere, with the farthest plane nearly dissolving",
    "a high viewpoint over an almost empty surface, a single small form and its softened reflection",
)
LIGHT = (
    "soft side light revealing one translucent edge; distant forms lose contrast gradually",
    "a narrow opening of warm light through cool haze, with controlled highlights and luminous shadows",
    "diffuse dawn light with one gently glowing reflection and subtle atmospheric color separation",
    "quiet backlighting and a faint halo around the focal silhouette, without clipping the whites",
)
SCENES = {
    "Surrealism": ("A single porcelain doorway stands in a shallow, still sea, its opening filled with distant dawn", "A Door for the Dawn"),
    "Impossible Architecture": ("A slender stair curves into pale mist above an empty courtyard, with no visible destination", "The Stair into Silence"),
    "Alien Worlds": ("One delicate mineral spire rises from a vast salt plain beneath a pale unfamiliar moon", "The Quiet Other Shore"),
    "Cosmic Art": ("A small crescent hangs above a barely visible planetary horizon in a vast luminous haze", "Between Two Horizons"),
    "Abstract Geometry": ("One translucent oval floats above a softly textured plane, casting a faint elongated shadow", "The Weight of an Oval"),
    "Crystal Structures": ("A single clear mineral blade emerges from smooth frost, with a tiny spectrum along its edge", "A Prism of Stillness"),
    "Futuristic Cities": ("A lone elevated pavilion glimmers through morning cloud above a distant, barely visible city", "The Pavilion Above Morning"),
    "Dreamlike Landscapes": ("A narrow island and one slender tree rest in an immense mirror lake, the far shore lost in mist", "The Island That Waits"),
    "Macro Photography": ("A single dew drop hangs from a fine curved stem against an open, softly luminous background", "A World Held Lightly"),
    "Experimental Fine Art": ("One sheet of translucent handmade paper curls above a pale stone surface, light passing through its fibers", "Light Through Paper"),
    "Liquid Metal": ("A thin ribbon of brushed silver lifts from a still reflective pool, with a quiet horizon behind it", "Silver Without Weight"),
    "Mathematical Fractals": ("A sparse branching glass structure fades into smaller translucent branches, surrounded by empty space", "The Last Branch of Light"),
    "Game Worlds": ("A small abandoned shrine rests on one floating terrace above a vast cloud sea; one distant path hints at a larger world", "The Unvisited Terrace"),
    "Storm Canyon": ("A narrow canyon opening frames a distant patch of luminous sky after rain, with a tiny stream far below", "When the Storm Becomes Quiet"),
    "Spring Snow, Warm Sun": ("A single flowering branch reaches over an untouched snowfield as warm dawn light touches the melting edge", "The Space Before Spring"),
}


def describe(theme, palette, composition, rng):
    subject, title = SCENES[theme]
    style = "ethereal fine-art composition"
    # Ask the official PUP to preserve the void instead of decorating it away.
    prompt = (
        f"A contemplative fine-art image. {subject}. Compose with {composition}. "
        f"Use {palette}; keep the strongest color local to the focal point. {rng.choice(LIGHT)}. "
        "Convey immense quiet and a sense of suspended time. Keep one clear visual anchor, "
        "subtle material detail near it, and soft edges toward the distance. Preserve substantial "
        "negative space and the stated scale relationships when expanding this description. "
        "Do not fill the empty areas with extra buildings, flowers, stars, ornaments or particles. "
        "Avoid uniform neon saturation, heavy bloom, busy spectacle, text, logos and game interfaces."
    )
    return subject, title, style, prompt
