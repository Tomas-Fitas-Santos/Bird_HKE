# Third-Party Notices

This repository includes third-party source code and derivative works.

## 1) MambaVision (NVIDIA)

- Source project: NVIDIA MambaVision repository
- Upstream folder in this repository: `MambaVision/`
- License file: `MambaVision/LICENSE`
- License type: NVIDIA Source Code License-NC (non-commercial)

### Scope

The license in `MambaVision/LICENSE` applies to files in `MambaVision/` and derivative files copied or adapted from that code.

### Obligations when redistributing

- Keep copyright notices
- Keep the full license text
- Keep attribution to the original project
- Respect non-commercial use terms from the NVIDIA Source Code License-NC

## 2) VHR-BirdPose

- Source project: VHR-BirdPose repository
- Upstream repository: https://github.com/LuoXishuang0712/VHR-BirdPose
- Scope in this repository: `Bird_HKE/models/VHR_BirdPose/` and any copied/adapted files derived from that branch

### Scope

The files in `Bird_HKE/models/VHR_BirdPose/` are adapted from the upstream VHR-BirdPose codebase and preserve upstream attribution headers.

That upstream project itself includes code derived from:

- HRNet / pose_hrnet from the Microsoft implementation under the MIT License
- ViTPose / OpenMMLab-derived backbone code and utilities

### Obligations when redistributing

- Keep the upstream file headers intact
- Keep any embedded third-party notices intact
- Preserve the upstream project attribution in the repository documentation
- Follow the license terms of the original components included by VHR-BirdPose

## 3) Project-specific code

Code outside third-party scopes (unless marked otherwise in file headers) is original code for this work.

If you add external code in the future, update this file with:

- project name and URL
- folder/file scope
- license type
- required attribution/redistribution terms

## 4) Recommended release checklist

Before making the repository public:

1. Ensure third-party folders keep their original LICENSE files.
2. Ensure copied/adapted files preserve relevant copyright headers.
3. Ensure README mentions third-party licensing constraints.
4. Remove model weights/data that cannot be redistributed.
5. Add citation references for upstream projects in your paper/repo docs.

## 5) Legal note

This document is a practical compliance note for research code release and is not legal advice.
