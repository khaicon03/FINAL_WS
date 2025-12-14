#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/chunnee/test/src/rosserial/rosserial_arduino"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/chunnee/test/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/chunnee/test/install/lib/python2.7/dist-packages:/home/chunnee/test/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/chunnee/test/build" \
    "/usr/bin/python2" \
    "/home/chunnee/test/src/rosserial/rosserial_arduino/setup.py" \
     \
    build --build-base "/home/chunnee/test/build/rosserial/rosserial_arduino" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/chunnee/test/install" --install-scripts="/home/chunnee/test/install/bin"
