from fastapi import FastAPI, WebSocket, WebSocketDisconnect,HTTPException
from fastapi.responses import FileResponse, JSONResponse
from WiseMind_doctor import process_single_disorder  # Ensure this is correctly imported
import os
import warnings
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
warnings.filterwarnings('ignore')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Serve index.html from the static folder
@app.get("/")
async def serve_index():
    file_path = os.path.join("static", "index.html")
    return FileResponse(file_path)

# Serve conversation.html from the static folder
@app.get("/conversation")
async def serve_conversation():
    file_path = os.path.join("static", "conversation.html")
    return FileResponse(file_path)

# Serve results.html from the static folder
@app.get("/results")
async def serve_results():
    file_path = os.path.join("static", "results.html")
    return FileResponse(file_path)

@app.get("/api/tree/{tree_type}")
async def get_decision_tree(tree_type: str):
    file_path = os.path.join("../data/Medical", f"{tree_type}_tree_dfs_demo.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Tree type not found")

    # Load CSV and filter only relevant columns (ignore 'Description')
    try:
        df = pd.read_csv(file_path, usecols=['path', 'name', 'parent','Description']).iloc[1:]
        tree_data = df.to_dict(orient="records")
        return JSONResponse(content=tree_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {e}")

# WebSocket endpoint to start the diagnostic process
@app.websocket("/ws/diagnostic")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")

    try:
        # Receive the initial setup data from the client
        data = await websocket.receive_json()
        print("Data received from client:", data)

        tree_type = data.get("tree_type")
        llm_type = data.get("llm_type")
        doctor_type = data.get("doctor_type")
        hci = data.get("hci", False)

        # Inform the client that the diagnostic process is starting
        # await websocket.send_text("Starting diagnostic process...")
        print("Starting diagnostic process...")

        # Set up the sample directory based on tree_type
        TEST_DIR = f"../data/Medical/mock_patient_{tree_type}/"

        # Run `process_single_disorder` asynchronously, passing the WebSocket
        result = await process_single_disorder(
            sample_dir=TEST_DIR,
            Disorder_Name="None",
            llm_type=llm_type,
            all_disorders="None",
            tree_type=tree_type,
            doctor_type=doctor_type,
            hci=hci,
            websocket=websocket
        )

        # Send the final diagnostic result to the clientdd
        await websocket.send_text(f"Diagnostic complete!\n Your result is : '{result['pred_disorder']}: {result['result_description']}'\nDiagnostic path: {result['details']['final_path']}")
        print("Diagnostic complete, result sent to client")

    except WebSocketDisconnect:
        print("Client disconnected from the WebSocket.")
    except Exception as e:
        print(f"Error during WebSocket communication: {e}")
        await websocket.send_text(f"An error occurred: {str(e)}")
    finally:
        print("WebSocket connection is now closing")
        await websocket.close()
        print("WebSocket connection closed")



