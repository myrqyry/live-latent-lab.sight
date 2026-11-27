ci_dependencies(){
  sudo apt-get install -y --no-install-recommends libopencv-dev libtesseract-dev
}
ci_build(){
  # Existing clean, model-free gate owns configuration, native tests, and
  # source/dependency provenance. Do not use locally installed models.
  SIGHT_OBS_SDK_ROOT="$CI_SDK" bash "$CI_ROOT/tools/verify-clean-build.sh" "$CI_BUILD"
}
ci_test(){
  # The gate above runs all configured tests. Verify its report again;
  # do not allow a local feature-rich build to masquerade as model-free.
  python3 - "$CI_BUILD/model-free.xml" "$MINIMUM_TESTS" <<'XML'
import sys,xml.etree.ElementTree as E
cases=E.parse(sys.argv[1]).findall('.//testcase')
assert len(cases)>=int(sys.argv[2])
assert all(c.find(k) is None for c in cases for k in ('failure','error','skipped'))
assert not any(c.get('status') in ('notrun','disabled') for c in cases)
print(f"Verified {len(cases)} model-free test cases")
XML
}
ci_stage(){
  cmake --install "$CI_BUILD" --prefix "$CI_STAGE/$MODULE"
  # The existing stage verifier checks the same install recipe, license
  # inventory and absence of model files. No artifact upload is enabled.
  bash "$CI_ROOT/tools/verify-release-stage.sh" "$CI_BUILD"
}
