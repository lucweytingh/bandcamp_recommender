# Instructions to Fix Import Issue in bandcamp_recommender

The issue is that `bandcamp_recommender` uses `src.recommendations` which conflicts with the local `src/recommendations` directory in `boemketel_radio`.

## Solution: Change the package structure to use a unique namespace

Execute these commands in `/Users/lucw/Documents/bandcamp_recommender`:

### Step 1: Rename the src directory to bandcamp_recommender
```bash
cd /Users/lucw/Documents/bandcamp_recommender
mv src bandcamp_recommender
```

### Step 2: Update pyproject.toml
Change the package name in `pyproject.toml`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["bandcamp_recommender"]
```

### Step 3: Update all imports in the package
Find and replace all imports from `from src.` or `from .` to use `bandcamp_recommender`:
```bash
# Update __init__.py files
sed -i '' 's/from \.recommendations/from bandcamp_recommender.recommendations/g' bandcamp_recommender/__init__.py
sed -i '' 's/from \.recommendations/from bandcamp_recommender.recommendations/g' bandcamp_recommender/recommendations/__init__.py

# Update all Python files in recommendations/
find bandcamp_recommender/recommendations -name "*.py" -exec sed -i '' 's/from \.\([a-z_]*\)/from bandcamp_recommender.recommendations.\1/g' {} \;
```

### Step 4: Reinstall the package
```bash
cd /Users/lucw/Documents/boemketel_radio
uv pip install -e /Users/lucw/Documents/bandcamp_recommender
```

### Step 5: Update imports in boemketel_radio
Change imports from `from src.recommendations` to `from bandcamp_recommender.recommendations`:
```bash
cd /Users/lucw/Documents/boemketel_radio
# Update bandcamp_suggestor.py
sed -i '' 's/from src\.recommendations import/from bandcamp_recommender.recommendations import/g' src/bandcamp_suggestor.py

# Update radio_generator.py  
sed -i '' 's/from src\.recommendations import/from bandcamp_recommender.recommendations import/g' src/radio_generator.py
```

### Step 6: Test
```bash
cd /Users/lucw/Documents/boemketel_radio
export PYTHONPATH=$(pwd)
uv run python -c "import src.radio_generator; print('✓ Import successful!')"
```

