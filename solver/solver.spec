# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['solver.py'],
    pathex=[],
    binaries=[('C:/Users/matte/Anaconda3/Lib/site-packages/pulp/solverdir/cbc/win/i64/cbc.exe', 'pulp/solverdir/cbc/win/i64/')],
    datas=[('tt.png', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='solver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['tt.png'],
)
