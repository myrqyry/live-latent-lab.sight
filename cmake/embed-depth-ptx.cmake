file(READ "${INPUT}" ptx)
file(WRITE "${OUTPUT}" "#pragma once\nstatic constexpr char sight_depth_ptx[] = R\"SIGHT_PTX(\n${ptx}\n)SIGHT_PTX\";\n")
