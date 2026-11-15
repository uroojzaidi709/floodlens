import pandas as pd
import streamlit as st

from common import ROOT, setup, demo_banner, na, get, load_manifest, check_uploaded_file, tamper_check, footer
from components.metric_cards import display_metric_card

event, S = setup("Forensic Audit", "🔐")

st.title("Data Provenance & Digital Forensics")
st.caption("Transparency, traceability and data integrity")
demo_banner(S)

f = S.get("forensics", {}) or {}
status = f.get("status")
if status == "CHECKS_PASSED":
    st.success("CHECKS PASSED", icon="✅")
else:
    st.warning(f"{status or 'status unavailable'}", icon="⚠️")
st.caption(f.get("disclaimer", "A matching hash proves the file was not altered. It does not prove the map is correct."))

g = f.get("geometry_audit", {}) or {}
st.subheader("Same viewpoint? (geometry check)")
c = st.columns(4)
with c[0]:
    display_metric_card("Orbit track", ", ".join(g.get("tracks_found", [])) or "n/a", "", "route", "Relative orbit")
with c[1]:
    display_metric_card("Direction", ", ".join(g.get("directions_found", [])).title() or "n/a", "", "explore",
                        "Flight direction")
with c[2]:
    display_metric_card("Days apart", g.get("days_between_pre_and_post", "n/a"), "days", "date_range",
                        "Pre to post scene")
with c[3]:
    display_metric_card("Result", "PASS" if g.get("passed") else "CHECK", "", "task_alt", "Same track and direction")
if g.get("note"):
    st.caption(g["note"])

try:
    M = load_manifest(event)
except Exception:
    M = None
    na("Provenance manifest not found.")

if M:
    st.divider()
    st.subheader("Where the data comes from")
    st.dataframe(pd.DataFrame(M["datasets"]).T[["name", "access", "role"]], use_container_width=True)

    st.subheader("File fingerprints (SHA-256)")
    hashes = M.get("artifact_sha256", {})
    st.dataframe(pd.DataFrame({"file": list(hashes), "sha256": list(hashes.values())}),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Tamper demo")
    st.markdown("Upload a file (for example the flood mask) and compare its fingerprint with the recorded one. "
                "Change one byte and it will not match.")
    usable = [k for k in hashes if not k.startswith("input_")]
    label = st.selectbox("Which file are you checking?", usable)
    up = st.file_uploader("Upload the file")
    if up is not None and label:
        r = check_uploaded_file(event, label, up.getvalue())
        (st.success if r["match"] else st.error)("MATCH: file is unchanged" if r["match"]
                                                 else "MISMATCH: file differs from the recorded one")
        st.code(f"expected {r['expected']}\nactual   {r['actual']}")
    st.caption("Input files (labels starting with input_) are not included in this deployment.")

    if st.button("Re-check files on disk"):
        try:
            t = tamper_check(event, base_dir=str(ROOT))
            st.write({"all_intact": t["all_intact"], "modified": t["modified"], "missing": t["missing"]})
        except Exception as e:
            st.warning(f"Could not re-check: {e}")

st.divider()
st.subheader("Processing lineage")
stages = ["Satellite data acquisition (Sentinel-1 via Earth Engine, NISAR via ASF)",
          "Same-orbit and metadata checks", "Preprocessing and change detection",
          "Terrain, permanent-water and speckle cleaning", "Exposure analysis (WorldPop, WorldCover)",
          "SHA-256 manifest and this app"]
for i, s in enumerate(stages, 1):
    with st.container(border=True):
        st.markdown(f"**{i}.** {s}")
st.caption("A matching hash proves the file was not altered. It does not prove the map is correct.")
footer()
