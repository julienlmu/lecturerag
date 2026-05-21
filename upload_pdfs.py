# Upload all local pdfs to Supabase

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

BUCKET = "lectures"
PDFS_DIR = Path("pdfs")

def get_remote_files(client) -> set[str]:
    """Return a set of all file paths in the bucket"""
    existing = set()
    folders = client.storage.from_(BUCKET).list()
    for entry in folders:
        if entry.get("metadata"):
            existing.add(entry["name"])
        else:
            folder_name = entry["name"]
            files = client.storage.from_(BUCKET).list(folder_name)
            for f in files:
                if f.get("metadata"):
                    existing.add(f"{folder_name}/{f['name']}")
    return existing

def upload_pdf(client, local_path: Path, remote_path: str) -> str:
    """Upload PDF and return its URl"""
    with open(local_path, "rb") as f:
        client.storage.from_(BUCKET).upload(
            path=remote_path,
            file=f,
            file_options={"content-type": "application/pdf"},
        )
    return client.storage.from_(BUCKET).get_public_url(remote_path)



def main(force: bool = False):
    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )


    if not PDFS_DIR.exists():
            print(f"ERROR: {PDFS_DIR} does not exist.")
            sys.exit(1)

    # Find every PDF in the pdfs/ folder, including subfolders
    local_pdfs = sorted(PDFS_DIR.rglob("*.pdf"))
    if not local_pdfs:
        print(f"No PDFs found in {PDFS_DIR}/")
        sys.exit(0)

    print(f"Found {len(local_pdfs)} local PDF(s).")

    # Fetch what's already in the bucket
    print("Checking remote bucket...")
    remote_files = get_remote_files(client)
    print(f"  {len(remote_files)} file(s) already in bucket.\n")

    uploaded = 0
    skipped = 0
    failed = 0

    for local_path in local_pdfs:
        # remote_path is the path relative to the pdfs/ folder

        remote_path = str(local_path.relative_to(PDFS_DIR))

        if remote_path in remote_files and not force:
            print(f"  SKIP   {remote_path} (already exists)")
            skipped += 1
            continue

        try:
            url = upload_pdf(client, local_path, remote_path)
            print(f"  UPLOAD {remote_path}")
            uploaded += 1
        except Exception as e:
            print(f"  FAIL   {remote_path}: {type(e).__name__}: {str(e)[:200]}")
            failed += 1

    print(f"\nDone. Uploaded: {uploaded}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)