# Patch MindSpore to add support for recompute in PyNative mode

# Find the site-packages path
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Define the file path and the line to insert
FILE_PATH="$SITE_PACKAGES/mindspore/common/recompute.py"
LINE_AFTER="        self.wrap_cell = _WrapCell(block)"
LINE_TO_INSERT="        self.wrap_cell.set_inputs()"

# Check if the file has already been modified
if grep -qF "$LINE_TO_INSERT" "$FILE_PATH"; then
    echo "File $FILE_PATH has already been patched. No changes made."
    exit 0
fi

# Use sed to insert the line after the specified pattern
if sed -i "/$LINE_AFTER/a \\$LINE_TO_INSERT" "$FILE_PATH"
then
    echo "Successfully patched $FILE_PATH"
else
    echo "Error: Failed to patch $FILE_PATH"
    exit 1
fi
