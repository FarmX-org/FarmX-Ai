def extract_soil_tag(soil_desc: str) -> str:
    soil_desc = soil_desc.lower()
    tags = ["sandy", "loam", "clay", "rocky", "silty", "peaty", "chalky"]
    for tag in tags:
        if tag in soil_desc:
            return tag
    return "other"
