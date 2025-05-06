# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import cv2
import google.generativeai as genai
import numpy as np
import os
from tempfile import NamedTemporaryFile
from typing import Dict, Any
from solve import process  # Import your processing function
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Parser import SpiceParser
import tempfile

app = FastAPI(
    title="Circuit Diagram Processor",
    description="API for processing circuit diagrams and generating netlists",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend IP
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")


@app.post("/image-to-netlist/")
async def image_to_netlist(file: UploadFile = File(...)):
    """Convert circuit image to netlist and solve it using AI model"""

    if file.content_type is None:
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and load image
        contents = await file.read()
        image_id = str(uuid.uuid4())
        path = f"images/{image_id}.png"
        with open(path, "wb") as f:
            f.write(contents)
        img = Image.open(io.BytesIO(contents))

        os.environ["CURRENT_IMAGE_PATH"] = path

        # Process the image - we'll need to modify your main() to return values
        netlist_part = await process()
        if not netlist_part:
            raise HTTPException(status_code=500, detail="Failed to generate netlist")
        voltages_part = solve_netlist(netlist_part)

        # Extract node voltages
        voltages = {}
        lines = voltages_part.splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                node, voltage = parts
                voltages[node] = float(voltage)
        print(netlist_part)
        print(voltages_part)
        return JSONResponse(
            content={"imageId": image_id, "netlist": netlist_part, "voltages": voltages}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/chat-about-circuit/")  # Chat Endpoint (Optional)
async def chat_about_circuit(
    question: Annotated[str, Form()],
    image_id: Annotated[str, Form()],
):
    """Chat with the AI about an uploaded circuit image."""
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="File must be an image")

    try:
        img_path = f"images/{image_id}.png"
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail="Image not found")
        # contents = await file.read()
        img = Image.open(img_path)
        # img = Image.open(io.BytesIO(contents))

        response = model.generate_content([question, img])

        if not response.text:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        return JSONResponse(content={"response": response.text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def solve_netlist(netlist: str):
    # Save netlist to a temp file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".cir", delete=False) as f:
        f.write(netlist)
        netlist_path = f.name

    # Parse and simulate the circuit
    parser = SpiceParser(path=netlist_path)
    circuit = parser.build_circuit()
    simulator = circuit.simulator()
    analysis = simulator.transient(step_time="1", end_time="500")

    # Print node voltages at last time point
    output = "Node\tVoltage\n"
    for node_name, node_voltage in analysis.nodes.items():
        output += f"{node_name}\t{float(node_voltage):.3f}\n"
    return output


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="172.105.41.70", port=8000)
