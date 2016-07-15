pushd docs
make html
ghp-import -n -p build/html/
popd
