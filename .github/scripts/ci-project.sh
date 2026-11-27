ci_dependencies(){
  command -v apt-get >/dev/null || { echo "Ubuntu qualification runner required" >&2; return 1; }
  sudo apt-get install -y --no-install-recommends libopencv-dev libtesseract-dev
}
ci_build(){
  # Existing clean, model-free gate owns configuration, native tests, and
  # source/dependency provenance. Do not use locally installed models.
  SIGHT_CI_OBS_VERSION="$OBS_VERSION" SIGHT_OBS_SDK_ROOT="$CI_SDK" bash "$CI_ROOT/tools/verify-clean-build.sh" "$CI_BUILD"
}
ci_test(){
  # The gate above runs all configured tests. Verify its report again;
  # do not allow a local feature-rich build to masquerade as model-free.
  python3 "$CI_ROOT/tools/verify-model-free-report.py" "$CI_BUILD/model-free.xml" "$MINIMUM_TESTS"
}
ci_stage(){
  cmake --install "$CI_BUILD" --prefix "$CI_STAGE/$MODULE"
  # The existing stage verifier checks the same install recipe, license
  # inventory and absence of model files before version-specific packaging.
  bash "$CI_ROOT/tools/verify-release-stage.sh" "$CI_BUILD"
  python3 "$CI_ROOT/tools/stage-ci-layout.py" "$CI_STAGE/$MODULE" "$CI_LAYOUT"
}
