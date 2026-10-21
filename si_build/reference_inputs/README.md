# reference_inputs

`captions/` holds one file per SI table: the text inside its `\caption{...}`, written by
the authors. `si_common.emit` splices it over the caption the table script wrote
(`SI_CAPTIONS_DIR` names another directory), so a regenerated table matches the published
one. To change a caption, edit its file here and re-run the table's script.
