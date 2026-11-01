"""Linux process-local seccomp guard; never changes host network configuration."""
import ctypes
import ctypes.util
import errno
import socket


def deny_internet_sockets():
    libname = ctypes.util.find_library('seccomp')
    if not libname:
        raise RuntimeError('libseccomp unavailable for kernel offline verification')
    lib = ctypes.CDLL(libname, use_errno=True)

    class Compare(ctypes.Structure):
        _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                    ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]

    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                          ctypes.c_uint, ctypes.POINTER(Compare)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    try:
        call = lib.seccomp_syscall_resolve_name(b'socket')
        for family in [socket.AF_INET, socket.AF_INET6]:
            comparison = Compare(0, 4, family, 0)  # argument 0, SCMP_CMP_EQ
            rc = lib.seccomp_rule_add_array(ctx, 0x00050000 | errno.EPERM, call, 1,
                                            ctypes.byref(comparison))
            if rc != 0:
                raise RuntimeError(f'seccomp_rule_add_array failed: {rc}')
        rc = lib.seccomp_load(ctx)
        if rc != 0:
            raise RuntimeError(f'seccomp_load failed: {rc}')
    finally:
        lib.seccomp_release(ctx)
    # Test via libc, bypassing Python-level guards; no connection is attempted.
    libc = ctypes.CDLL(None, use_errno=True)
    probes = []
    for family in [socket.AF_INET, socket.AF_INET6]:
        fd = libc.socket(family, socket.SOCK_STREAM, 0)
        error = ctypes.get_errno()
        if fd >= 0:
            libc.close(fd)
            raise RuntimeError('Kernel internet socket guard ineffective')
        if error != errno.EPERM:
            raise RuntimeError(f'Unexpected guard probe errno: {error}')
        probes.append({'family': int(family), 'returncode': fd, 'errno': error})
    return {'mechanism': 'seccomp: deny socket(AF_INET/AF_INET6) for process and descendants',
            'libc_socket_probes': probes}
