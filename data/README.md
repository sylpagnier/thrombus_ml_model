# Data Directory

This directory contains the raw and processed data for the ML CFD thrombus predictions project.

## Structure

- `raw/` - Raw input data files (e.g., .nas mesh files from CT images)
- `processed/` - Processed/preprocessed data (to be created during Stage 2)

## Importing .nas Files

### Automatic Import from Downloads

Use the utility script to automatically import `.nas` files from your Downloads folder:

```powershell
python scripts/move_nas_files.py "C:\Users\pgssy\Downloads" --copy
```

The script will:
- Find all `.nas` files in the specified directory (including subdirectories)
- Copy them to `data/raw/`
- Handle name conflicts automatically by appending numbers

### Manual Import

You can also manually copy or move `.nas` files:

```powershell
# Copy files
Copy-Item "C:\path\to\your\*.nas" -Destination "data\raw\"

# Or move files
Move-Item "C:\path\to\your\*.nas" -Destination "data\raw\"
```

## File Management

Large `.nas` files are automatically excluded from:
- **Version control** (via `.gitignore`) - prevents committing large files
- **AI tool reading** (via `.cursorignore`) - improves development performance

This ensures these large mesh files won't slow down development or accidentally be committed to the repository.
