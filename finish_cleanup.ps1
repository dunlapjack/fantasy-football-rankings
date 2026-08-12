# Finishes the Phase 13.5 cleanup and pushes.
#
# WHY THIS IS A SCRIPT INSTEAD OF ALREADY DONE. The sandbox that did the
# analysis has no unlink permission inside the OneDrive mount and no
# network route to GitHub (proxy returns 403 on CONNECT). It could create
# git objects and commit, but not delete a file or push. Everything that
# could be done there was; this is the remainder.
#
#   Run from the repo root:   .\finish_cleanup.ps1

$ErrorActionPreference = "Stop"

# 1. A stale lock is left behind because the sandbox could create
#    .git/index.lock but not remove it. Git refuses every command until
#    it is gone.
if (Test-Path .git\index.lock) {
    Remove-Item .git\index.lock -Force
    Write-Host "removed stale .git/index.lock"
}

# 2. The Phase 13.5 commit was made against an out-of-tree index, so
#    .git/index still describes the pre-commit state. Resync it to HEAD.
#    This touches no working file.
git reset -q
Write-Host "index resynced to HEAD"

# 3. Drop the superseded v13 boards from tracking. They stay recoverable
#    at 53d8314, per the rule already written into .gitignore: draft
#    boards are build artifacts.
git rm -q --cached 2026_12Team_Board_v13.xlsx 2026_32Team_Board_v13.xlsx 2026_6Team_Board_v13.xlsx

# 4. Delete the superseded working files.
#
#    SAFE, and each for a stated reason:
#      v13 boards          tracked at 53d8314, recoverable
#      boards_v13_frozen/  verified byte-identical to those (md5)
#      verify/             wiring-check builds; regenerable with
#                          build_board --output, and their result is
#                          recorded in PHASE_8-14_PLAN.md
#      v14 boards          never committed, so this IS permanent -- but
#                          v15 was verified rank-identical to v14 and
#                          differs only in Exp Gm/Exp Pts for 79 rookies,
#                          where v15 is the corrected one
$doomed = @(
    "2026_12Team_Board_v13.xlsx", "2026_32Team_Board_v13.xlsx", "2026_6Team_Board_v13.xlsx",
    "2026_12Team_Board_v14.xlsx", "2026_32Team_Board_v14.xlsx", "2026_6Team_Board_v14.xlsx",
    "verify", "boards_v13_frozen"
)
foreach ($item in $doomed) {
    if (Test-Path $item) {
        Remove-Item $item -Recurse -Force
        Write-Host "deleted $item"
    }
}

# LibreOffice lock file -- appears when QB59_stress_test.xlsx is open.
Get-ChildItem -Filter ".~lock.*#" -Force -ErrorAction SilentlyContinue | Remove-Item -Force

# 5. Commit the cleanup separately from the phase, so a revert of either
#    does not drag the other with it.
git add -A
git commit -q -m @"
Cleanup: drop superseded v13 and v14 boards

Draft boards are build artifacts, per the rule already in .gitignore:
'delete old versions freely. Recover any previously committed one with
git checkout <commit> -- <file>'. v13 remains recoverable at 53d8314.

Superseded rather than equivalent, and shown rather than assumed: v13 to
v15 moves 878 of 1082 ranks, 38 of them inside the top 120, because the
Aug 12 data refresh sits between them. v14 to v15 was verified
rank-identical and differs only in Exp Gm/Exp Pts for 79 rookies.

Also removes verify/ (wiring-check builds, regenerable, result recorded
in the plan) and boards_v13_frozen/ (verified byte-identical to the
tracked v13 boards before deletion).
"@

Write-Host ""
Write-Host "=== pushing ==="
git push origin main

Write-Host ""
Write-Host "Done. Remaining boards:"
Get-ChildItem *.xlsx | Select-Object -ExpandProperty Name
