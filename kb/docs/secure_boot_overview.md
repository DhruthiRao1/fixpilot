---
title: Secure Boot - what it does and when to change it
vendor: any
topic: secure_boot
---
Secure Boot makes the firmware verify the signature of the bootloader before running it,
blocking bootkits and other pre-OS malware. Windows 11 expects Secure Boot to be enabled.
Windows installation media and many Linux distributions (Ubuntu, Fedora) are signed and boot
with Secure Boot on. Some repair tools and unsigned distributions require it to be disabled
temporarily. Best practice: only disable it for a specific need and re-enable it afterwards.
Changing Secure Boot can change TPM measurements and trigger BitLocker recovery on the next
start, so confirm the recovery key is available first.
