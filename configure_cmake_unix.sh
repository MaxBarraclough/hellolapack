

echo "REMEMBER THIS USES DEFAULT CONFIG. Neither RELEASE nor DEV nor DEBUG"



## First, cd into the directory containing this script
cd `dirname "$(readlink -f "$0")"`


mkdir 'build'

cd build

cmake -G "Unix Makefiles" ../src/



## Little reason to use this newer syntax:

# cmake               \
# -G "Unix Makefiles" \
# -S ./src            \
# -B ./build

