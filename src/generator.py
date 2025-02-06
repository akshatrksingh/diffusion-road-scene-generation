import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import spacy
import logging
import numpy as np
from typing import Optional
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RoadSceneGenerator:
    """
    Generates road scene images based on text descriptions.
    Uses a diffusion model (Kandinsky) and NLP (spaCy) for enhancements.
    """
    def __init__(self):
        self.device = torch.device('cpu')
        self.nlp = spacy.load("en_core_web_sm")

        try:
            self.prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-prior",
                torch_dtype=torch.float32,
            ).to(self.device)

            self.pipe = KandinskyV22Pipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder",
                torch_dtype=torch.float32,
            ).to(self.device)

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def enhance_description(self, description: str) -> str:
        """
        Enhances a text description by adding scene-specific details.
        """
        doc = self.nlp(description)

        scene_features = {
            "highway": ["clear lane markings", "overhead signs", "multiple lanes"],
            "city_road": ["synchronized traffic signals", "bright billboards", "marked crosswalks"],
            "rural_road": ["natural roadside vegetation", "curvy paths", "scenic landscapes"]
        }

        scene_type = self._detect_scene_type(doc)

        if scene_type in scene_features:
            rng = np.random.default_rng()
            features = scene_features[scene_type]
            selected_features = rng.choice(features, size=2, replace=False)
            description += f", featuring {', '.join(selected_features)}"

        city_name = self._detect_location(doc)
        if city_name:
            description += f". This scene is set in {city_name}, known for its unique cityscape."

        return description

    def _detect_scene_type(self, doc) -> str:
        """
        Detects the type of scene based on keywords in the input text.
        """
        scene_keywords = {
            "highway": {"highway", "freeway", "interstate"},
            "city_road": {"city", "urban", "street"},
            "rural_road": {"rural", "countryside", "village"}
        }

        text_lower = doc.text.lower()
        for scene_type, keywords in scene_keywords.items():
            if keywords.intersection(text_lower.split()):
                return scene_type

        return "general_road"

    def _detect_location(self, doc) -> Optional[str]:
        """
        Detects a city or location mentioned in the input text.
        """
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC"}:
                return ent.text
        return None

    def generate_scene(self, 
                       description: str, 
                       seed: Optional[int] = None,
                       negative_prompt: str = "low quality, blurry, noisy, animated",
                       ) -> Image.Image:
        """
        Generates an image based on the provided description.
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        enhanced_description = self.enhance_description(description)
        logger.info(f"Processing enhanced description: {enhanced_description}")

        try:
            embeds = self.prior_pipeline(
                prompt=enhanced_description,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                generator=generator
            )

            image = self.pipe(
                image_embeds=embeds.image_embeds,
                negative_image_embeds=embeds.negative_image_embeds,
                guidance_scale=7.0,
                num_inference_steps=30,
                height=512,
                width=512
            ).images[0]

            return image

        except Exception as e:
            logger.error(f"Error in scene generation: {str(e)}")
            raise
