---
title: Creating bootable USB media
vendor: any
topic: boot
---
A USB stick only appears in the boot menu if it contains a UEFI boot loader at
EFI/BOOT/BOOTX64.EFI. Copying an ISO file onto the drive does not make it bootable. Use
Microsoft's Media Creation Tool for Windows, or Rufus with partition scheme GPT and target
system UEFI. Prefer a rear/motherboard USB port and avoid hubs. If the drive still does not
appear, try "Boot From EFI File" and browse to EFI/BOOT/BOOTX64.EFI.
