"""Tiny advice translation dict (not a general translator)."""

EN_HI = {
    "Decision support only. Not a certified agronomic prescription.": "केवल निर्णय सहायता। प्रमाणित कृषि सलाह नहीं है।",
    "Typical-range check": "सामान्य सीमा जाँच",
    "This is an association in the model, not a proven cause.": "यह मॉडल में एक संबंध है, सिद्ध कारण नहीं।",
    "Planner is decision support only. Not a certified agronomic prescription. "
    "Fertilizer search is a model association inside the training range; irrigation is an FAO-56 calculation.": "योजनाकार केवल निर्णय सहायता है। प्रमाणित कृषि नुस्खा नहीं। "
    "उर्वरक खोज प्रशिक्षण सीमा में मॉडल संबंध है; सिंचाई FAO-56 गणना है।",
}


def t(text: str, lang: str) -> str:
    if lang == "hi":
        return EN_HI.get(text, text)
    return text
