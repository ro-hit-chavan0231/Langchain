from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field


model = ChatOllama(model = 'qwen2.5-coder:7b')

class Review(BaseModel):

    key_themes : list[str] = Field(description='Write down all the key themes discussed in the list')
    summary : str = Field(description="A brief Summary of the review")
    sentiment : Literal['Pos', "Neg"] = Field(description="Return a Sentiment of the review either negative, positive or neutral")
    pros : Optional[list[str]] = Field(default=None, description='Write down all the pros inside the list')
    cons : Optional[list[str]] = Field(default=None, description='Write down all the cons inside the list') 
    name : Optional[str] = Field(description= 'Write the name of person who write the review')
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I had a very disappointing experience with this clothing brand and would not recommend it based on my purchase. The clothes looked stylish in the product photos, but the actual items were nothing like what was advertised. The fabric felt extremely cheap, rough, and uncomfortable to wear. After wearing the shirt just once and washing it according to the care instructions, the color started fading and the material lost its shape.

The sizing was also completely inaccurate. I ordered my usual size based on the size chart, but the fit was either too tight in some areas or too loose in others. It was clear that quality control was lacking. Even the stitching was poorly done, with loose threads visible right out of the package.

To make matters worse, the delivery took much longer than promised. The packaging was damaged when it arrived, which gave a poor first impression. I contacted customer support regarding the quality issues and sizing problem, but the response was slow and not very helpful. The return process was unnecessarily complicated, and I felt like my concerns were not taken seriously.

Considering the price, I expected much better quality and service. There are many other brands offering better fabrics, more accurate sizing, faster delivery, and responsive customer support at a similar or even lower price. Unfortunately, this purchase was a complete waste of money, and I won't be shopping from this brand again.
""")

print(result.key_themes)