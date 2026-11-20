---
title: BitLocker / Device Encryption recovery key
vendor: any
topic: encryption
---
Windows asks for the 48-digit BitLocker recovery key when it detects a change in boot
configuration, such as Secure Boot or TPM settings changing. Personal devices signed in with
a Microsoft account store the key at aka.ms/myrecoverykey; match the Key ID shown on screen.
Work devices store keys in Entra ID / Active Directory, retrievable by IT.

To check whether a drive is encrypted: File Explorer -> This PC shows a padlock on the C:
drive when encrypted; or run "manage-bde -status C:" as administrator and look for
"Protection On". If Settings -> Privacy & security has no Device encryption entry and no
padlock is shown, the drive is not encrypted. Before changing firmware security settings,
suspend protection from Manage BitLocker or record the key with
"manage-bde -protectors -get C:".
