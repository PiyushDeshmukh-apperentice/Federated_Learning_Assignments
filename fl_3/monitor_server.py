from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
import datetime

app = FastAPI(title="Federated Learning Monitoring System")

# In-memory database to store updates from different clients
client_updates = {}

@app.post("/receive_update")
async def receive_update(request: Request):
    """Endpoint for clients to transmit their local model updates."""
    data = await request.json()
    partition_id = data.get("partition_id")
    
    # Store the update with a timestamp
    client_updates[partition_id] = {
        "weights": data.get("weights"),
        "accuracy": data.get("accuracy"),
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    }
    
    print(f"📥 [Server] Received update from Client {partition_id}")
    return {"status": "success", "message": "Update logged by Monitoring System"}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Simple HTML Dashboard to view the transmitted updates."""
    rows = ""
    for cid, info in sorted(client_updates.items()):
        # Truncate weights for display
        weight_snippet = str(info['weights'][0][:5]) + "..." 
        rows += f"""
        <tr>
            <td>{cid}</td>
            <td>{info['timestamp']}</td>
            <td>{info['accuracy']:.4f}</td>
            <td style="font-family: monospace; font-size: 12px;">{weight_snippet}</td>
            <td style="color: green;">Transmitted</td>
        </tr>
        """

    html_content = f"""
    <html>
        <head>
            <title>FL Monitor</title>
            <style>
                body {{ font-family: sans-serif; margin: 40px; background: #f4f4f9; }}
                table {{ width: 100%; border-collapse: collapse; background: white; }}
                th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>Centralized FL Update Monitor (Assignment 3)</h2>
            <p>This server simulates the central endpoint receiving local model updates.</p>
            <table>
                <tr>
                    <th>Client ID</th>
                    <th>Last Update</th>
                    <th>Local Accuracy</th>
                    <th>Weight Preview (Coefficients)</th>
                    <th>Status</th>
                </tr>
                {rows if rows else "<tr><td colspan='5'>Waiting for client updates... Run 'flwr run .' to start.</td></tr>"}
            </table>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)