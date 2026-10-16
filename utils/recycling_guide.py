"""
Domain Knowledge Engine for Waste Classification and Recycling.
Provides structured guidelines, disposal steps, municipal bin codes,
carbon/energy savings data, and decomposition estimates for each category.
"""

RECYCLING_GUIDELINES = {
    "plastic": {
        "title": "Plastic Waste (PET, HDPE, PP)",
        "icon": "🧴",
        "category_badge": "Recyclable Polymer",
        "recyclability_status": "Recyclable (Codes 1, 2, 5)",
        "bin_color": "Yellow Bin (Clean Dry Plastics)",
        "bin_hex": "#FBC02D",
        "prep_steps": [
            "Empty any remaining liquid or food residue completely.",
            "Rinse with clean or greywater to prevent contamination.",
            "Remove plastic caps and label films if required by local guidelines.",
            "Crush or flatten bottles to save volume in recycling bins.",
            "Keep dry and separate from hazardous chemical containers."
        ],
        "dos": [
            "Recycle clean beverage bottles (PET #1) and milk jugs (HDPE #2).",
            "Recycle clean yogurt tubs and takeaway containers (PP #5).",
            "Crush bottles vertically before replacing the cap."
        ],
        "donts": [
            "Do NOT recycle plastic bags, cling wraps, or bubble wraps in curbside bins (they jam sorting machines).",
            "Do NOT include greasy food-stained plastic containers (e.g., oily salad bowls).",
            "Do NOT mix medical syringes, bioplastics (PLA), or hazardous chemical bottles."
        ],
        "decomposition_time": "450 to 1,000 years in landfill",
        "environmental_impact": {
            "energy_saved_per_ton": "5,774 kWh of electricity",
            "oil_saved_per_ton": "16.3 barrels (2,600 liters) of crude oil",
            "co2_reduction_per_ton": "1.5 tons of CO2 emissions avoided",
            "water_saved_per_ton": "25,000 liters of fresh water"
        },
        "upcycling_ideas": [
            "Convert 2-liter soda bottles into self-watering indoor plant pots.",
            "Cut plastic jugs into garden scoops or bird feeders.",
            "Use milk jugs as frost protectors for delicate seedling plants."
        ],
        "common_items": "Water bottles, soda bottles, shampoo bottles, milk jugs, food tubs, detergent bottles."
    },

    "paper": {
        "title": "Clean Paper & Newsprint",
        "icon": "📄",
        "category_badge": "Cellulose Fiber",
        "recyclability_status": "100% Recyclable (Up to 5-7 times)",
        "bin_color": "Blue Bin (Paper & Fiber Stream)",
        "bin_hex": "#1565C0",
        "prep_steps": [
            "Ensure the paper is completely dry and clean from oils or grease.",
            "Remove plastic binder covers, laminated sheets, and metallic clips.",
            "Flatten newspapers, office sheets, and envelopes.",
            "Do not shred unless containing confidential data (shredded fibers are too short for high-grade recycling)."
        ],
        "dos": [
            "Recycle office copy paper, magazines, brochures, and newspapers.",
            "Recycle paper grocery bags and envelopes (including clear window envelopes).",
            "Keep in a dry indoor spot until bin collection day."
        ],
        "donts": [
            "Do NOT recycle greasy pizza boxes, wax paper, or thermal cash receipts (BPA coated).",
            "Do NOT recycle paper towels, facial tissues, or wet napkins (soiled fibers cannot be reused).",
            "Do NOT mix foil-embossed or heavily plastic-laminated wrapping paper."
        ],
        "decomposition_time": "2 to 6 weeks (if clean and moist)",
        "environmental_impact": {
            "energy_saved_per_ton": "4,100 kWh of electricity (saves 60% vs virgin pulp)",
            "oil_saved_per_ton": "17 trees preserved per ton of paper recycled",
            "co2_reduction_per_ton": "1.3 tons of CO2 equivalent",
            "water_saved_per_ton": "26,500 liters of water conserved"
        },
        "upcycling_ideas": [
            "Use newspaper sheets as compost carbon layers (browns) in garden compost.",
            "Fold old flyers and magazines into gift boxes or paper mache crafts.",
            "Use shredded non-glossy paper as eco-friendly packaging cushion material."
        ],
        "common_items": "Notebooks, office paper, envelopes, catalogs, newspapers, cardboard sleeves."
    },

    "cardboard": {
        "title": "Corrugated & Solid Cardboard",
        "icon": "📦",
        "category_badge": "Heavy Fiberboard",
        "recyclability_status": "Highly Recyclable (High Economic Value)",
        "bin_color": "Blue Bin (Paper & Fiberboard)",
        "bin_hex": "#1E88E5",
        "prep_steps": [
            "Remove all plastic packaging materials, bubble wrap, and styrofoam inserts.",
            "Remove heavy packing tape and shipping shipping labels where feasible.",
            "Break down and flatten all boxes completely to conserve bin capacity.",
            "Keep dry — wet cardboard easily molds and degrades pulp fibers."
        ],
        "dos": [
            "Flatten shipping cartons, cereal boxes, shoe boxes, and egg cartons.",
            "Cut oversized boxes into manageable 2x2 ft pieces for curbside pickup.",
            "Ensure boxes are clean and free from grease or chemical spills."
        ],
        "donts": [
            "Do NOT recycle greasy, sauce-soaked pizza box bottoms (tear off clean lids instead!).",
            "Do NOT include wax-coated produce boxes or frozen food boxes with plastic lining.",
            "Do NOT allow cardboard to sit outside in rain before recycling."
        ],
        "decomposition_time": "2 months (under optimal composting conditions)",
        "environmental_impact": {
            "energy_saved_per_ton": "4,000 kWh of electrical energy",
            "oil_saved_per_ton": "390 gallons (1,475 liters) of oil",
            "co2_reduction_per_ton": "9 cubic yards of landfill space preserved",
            "water_saved_per_ton": "7,000 gallons of fresh water"
        },
        "upcycling_ideas": [
            "Use flattened corrugated cardboard as weed barriers under garden mulch.",
            "Construct cat scratching pads by rolling up glued corrugated strips.",
            "Use sturdy shipping boxes as modular closet storage organizers."
        ],
        "common_items": "Shipping boxes, cereal boxes, shoe boxes, corrugated packaging, egg cartons."
    },

    "glass": {
        "title": "Glass Bottles & Jars",
        "icon": "🍾",
        "category_badge": "Inorganic Silica",
        "recyclability_status": "Infinitely Recyclable (100% Closed Loop)",
        "bin_color": "Green Bin (Glass Packaging)",
        "bin_hex": "#2E7D32",
        "prep_steps": [
            "Empty contents and lightly rinse out food/drink residue.",
            "Remove metal or plastic screw caps and corks (recycle caps separately).",
            "Paper labels do NOT need to be removed (burns off in furnace).",
            "Do not intentionally break bottles; unbroken glass is safer for sanitation workers."
        ],
        "dos": [
            "Recycle all clear, brown, and green beverage bottles.",
            "Recycle condiment jars (jam, pasta sauce, baby food, olive oil).",
            "Sort by color if local community provides separate color-coded glass bins."
        ],
        "donts": [
            "Do NOT mix drinking glasses, wine glasses, or crystal (contain lead/differing melting temps).",
            "Do NOT mix Pyrex, ovenware, or ceramics (ruins the molten glass batch).",
            "Do NOT include window panes, mirrors, or incandescent light bulbs."
        ],
        "decomposition_time": "Over 1,000,000 years (practically immortal in landfill)",
        "environmental_impact": {
            "energy_saved_per_ton": "42 kWh of energy per ton for every 10% cullet used",
            "oil_saved_per_ton": "1.2 tons of virgin raw materials (sand, soda ash, limestone) saved",
            "co2_reduction_per_ton": "315 kg of direct CO2 emissions prevented per ton",
            "water_saved_per_ton": "50% reduction in water pollution compared to new glass"
        },
        "upcycling_ideas": [
            "Repurpose mason jars for bulk pantry storage (spices, grains, legumes).",
            "Turn wine and beer bottles into decorative fairy light lamps or candle holders.",
            "Use wide-mouth jars for indoor water propagation of house plants."
        ],
        "common_items": "Beer bottles, wine bottles, soda bottles, jam jars, pickle jars, sauce containers."
    },

    "metal": {
        "title": "Metal & Aluminum Cans",
        "icon": "🥫",
        "category_badge": "Ferrous & Non-Ferrous Metals",
        "recyclability_status": "Infinitely Recyclable (Saves 95% Energy)",
        "bin_color": "Silver / Gray Bin (Metals Stream)",
        "bin_hex": "#78909C",
        "prep_steps": [
            "Empty liquids or scrape food remains completely.",
            "Rinse clean to avoid attracting pests or foul odors.",
            "Aluminum drink cans can be crushed flat to save bin space.",
            "Tuck the tin can lid inside the can to avoid exposed sharp edges."
        ],
        "dos": [
            "Recycle soda and beer aluminum beverage cans.",
            "Recycle tin/steel food cans (soup, beans, tuna, pet food).",
            "Recycle clean, ball-shaped aluminum foil (minimum tennis-ball size)."
        ],
        "donts": [
            "Do NOT recycle paint cans, motor oil cans, or pesticide containers (hazardous).",
            "Do NOT place aerosol cans that are not completely empty into regular recycling.",
            "Do NOT include electronic parts, wires, batteries, or scrap iron in curbside bins."
        ],
        "decomposition_time": "200 to 500 years for aluminum; 50 to 100 years for tin/steel",
        "environmental_impact": {
            "energy_saved_per_ton": "14,000 kWh of electricity (saves 95% vs bauxite mining)",
            "oil_saved_per_ton": "40 barrels of oil equivalent",
            "co2_reduction_per_ton": "9 tons of CO2 greenhouse gases avoided per ton of aluminum",
            "water_saved_per_ton": "97% reduction in water pollution vs primary smelting"
        },
        "upcycling_ideas": [
            "Decorate cleaned tin cans into pencil holders or utensil organizers.",
            "Punch patterns into steel cans to create tea light candle lanterns.",
            "Use small tins as succulent mini-planters with drainage holes drilled in bottom."
        ],
        "common_items": "Soda cans, beer cans, soup tins, canned fruit/vegetables, clean foil trays, metal jar lids."
    },

    "trash": {
        "title": "Organic / General Municipal Solid Waste",
        "icon": "🍎",
        "category_badge": "Organic & Residual Waste",
        "recyclability_status": "Compostable (Organic) or Non-Recyclable Residual",
        "bin_color": "Black / Brown Bin (Compost or Landfill)",
        "bin_hex": "#424242",
        "prep_steps": [
            "Separate organic food scraps from inorganic plastic wraps or packaging.",
            "Place organic biodegradable waste into the green/brown compost bin.",
            "Ensure no non-compostable plastics or metals are mixed with compost.",
            "Wrap non-recyclable hygiene waste securely before landfill disposal."
        ],
        "dos": [
            "Compost fruit peels, vegetable scraps, coffee grounds, and tea leaves.",
            "Compost dry leaves, garden clippings, and non-printed cardboard shreds.",
            "Dispose of mixed multi-layer packaging (crisp packets, dirty plastic wraps) in black landfill bin."
        ],
        "donts": [
            "Do NOT put meat, dairy, or cooked oily food in small home compost bins (attracts rodents).",
            "Do NOT put hazardous waste (e.g., batteries, light bulbs, paint) into the trash bin.",
            "Do NOT bag recyclables inside black garbage bags (workers cannot inspect them)."
        ],
        "decomposition_time": "1 to 6 months for compost; up to 25+ years if starved of oxygen in deep landfill",
        "environmental_impact": {
            "energy_saved_per_ton": "Composting diverts organic matter from generating methane gas (CH4)",
            "oil_saved_per_ton": "Methane is 28x more potent than CO2 as a greenhouse gas",
            "co2_reduction_per_ton": "Produces rich organic fertilizer reducing synthetic chemical fertilizers",
            "water_saved_per_ton": "Compost enhances soil moisture retention by up to 30%"
        },
        "upcycling_ideas": [
            "Start a backyard compost heap or kitchen Bokashi bin for kitchen scraps.",
            "Use coffee grounds directly as nitrogen-rich garden fertilizer for acid-loving plants.",
            "Re-grow vegetable scraps (scallions, celery, romaine lettuce) in shallow water bowls."
        ],
        "common_items": "Fruit peels, food leftovers, soiled napkins, broken ceramics, mixed composite packaging."
    }
}


def get_guideline(waste_class: str) -> dict:
    """Safely retrieves recycling guideline for any of the 6 classes."""
    key = waste_class.lower().strip()
    if key in RECYCLING_GUIDELINES:
        return RECYCLING_GUIDELINES[key]
    return RECYCLING_GUIDELINES["trash"]


def calculate_eco_savings(waste_class: str, item_count: int = 1) -> dict:
    """Calculates approximate ecological savings for user gamification/impact tracking."""
    # Unit conversions per item (approximate based on standard weights)
    weights_kg = {
        "plastic": 0.025,   # ~25g per bottle
        "paper": 0.010,     # ~10g per sheet/page
        "cardboard": 0.150, # ~150g per small box
        "glass": 0.350,     # ~350g per glass bottle
        "metal": 0.015,     # ~15g per aluminum can
        "trash": 0.100      # ~100g organic waste diverted to compost
    }
    weight = weights_kg.get(waste_class.lower(), 0.05) * max(1, item_count)

    # Multipliers per kg recycled
    co2_kg = weight * 1.8
    energy_kwh = weight * 4.2
    water_liters = weight * 12.0

    return {
        "items": item_count,
        "weight_kg": round(weight, 3),
        "co2_saved_kg": round(co2_kg, 2),
        "energy_saved_kwh": round(energy_kwh, 2),
        "water_saved_liters": round(water_liters, 1)
    }
