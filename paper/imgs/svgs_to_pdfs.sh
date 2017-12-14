#!/usr/bin/env bash

for f in *.svg; do
  inkscape --without-gui --export-pdf="${f%.*}.pdf" "$f"
done
