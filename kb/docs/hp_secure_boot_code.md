---
title: HP Operating System Boot Mode Change confirmation code
vendor: HP
topic: secure_boot
---
After Secure Boot is enabled or disabled and changes are saved, HP firmware shows
"Operating System Boot Mode Change (021)" on the next start and asks the user to type a
4-digit pass code followed by Enter. The digits are not echoed on screen. Typing the code
incorrectly does nothing; check each digit and retry. Pressing Esc continues without
applying the pending change. The code exists to stop malware or another person from
changing Secure Boot silently.
