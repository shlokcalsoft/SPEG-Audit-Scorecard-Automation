from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
import subprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
TEMPLATE_FILE = PROJECT_ROOT / "template" / "SEPG_Master_Audit_Template.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "data" / "SEPG_Master_Audit_Completed.xlsx"
REFERENCE_DIR = PROJECT_ROOT / "reference"

def run_pipeline():
    try:
        print("Running transcript_parser...")
        subprocess.run(["python", "-m", "src.transcript_parser"], check=True)
        print("Running retriever...")
        subprocess.run(["python", "-m", "src.retriever"], check=True)
        print("Running evidence_extractor...")
        subprocess.run(["python", "-m", "src.evidence_extractor"], check=True)
        print("Running assessment_engine...")
        subprocess.run(["python", "-m", "src.assessment_engine"], check=True)
        print("Running excel_exporter...")
        subprocess.run(["python", "-m", "src.excel_exporter"], check=True)
        print("Pipeline finished.")
    except subprocess.CalledProcessError as e:
        print(f"Error in pipeline: {e}")
        raise

@app.post("/api/upload")
async def upload_transcript(
    file: UploadFile = File(...),
    project: str = Form(...)
):
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are allowed")

    # If user selected a project, copy it to the template location
    if project and project != "default":
        project_file = REFERENCE_DIR / project
        if project_file.exists():
            shutil.copy2(project_file, TEMPLATE_FILE)
        else:
            raise HTTPException(status_code=400, detail="Selected project file not found")

    # Clear transcript dir
    if TRANSCRIPT_DIR.exists():
        for f in TRANSCRIPT_DIR.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except:
                    pass
    else:
        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # Save new transcript
    file_path = TRANSCRIPT_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run pipeline
    try:
        run_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(
        path=OUTPUT_FILE,
        filename=f"{file.filename.split('.')[0]}_Audit_Completed.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/api/projects")
async def list_projects():
    projects = []
    
    if REFERENCE_DIR.exists():
        for f in REFERENCE_DIR.glob("*.xlsx"):
            projects.append({"name": f.name, "folder": "reference"})
            
    # Also add default template
    if TEMPLATE_FILE.exists():
        projects.insert(0, {"name": "Default SEPG Master Audit Template", "folder": "template", "id": "default"})
            
    return {"projects": projects}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
