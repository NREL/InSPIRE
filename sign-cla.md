Sign the CLA
=============

This page is the step-by-step guide to signing the Alliance's 
Contributors License Agreement. It's easy and pretty painless!

1. First and foremost, read [the current version of the
   CLA](cla-1.0.md). It is written to be as close to plain
   English as possible.

2. Make an account on [GitHub](https://github.com/) if you don't already
   have one.

3. File a pull request on this project (the InSPIRE Project), as [outlined below](#filing-the-pull-request).

4. Email the InSPIRE Sourceror, as [outlined below](#sending-the-email).

5. Wait for an Alliance team member to merge your pull request. You may start
   opening pull requests for the project you're contributing to but we will
   only be able to merge your contributions after your signed CLA is merged.


* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Filing the Pull Request
-----------------------

If you don't yet know how to file a pull request, read [GitHub's
document about it](https://help.github.com/articles/using-pull-requests).

Make your pull request be the addition of a single file to the
[contributors](contributors) directory of this project. Name the file
with the same name as your GitHub userid, with `.md` appended to the
end. For example, for the user `shirubana`, the full path to the file
would be `contributors/shirubana.md`.

Put the following in the file:

```
[date]

I hereby agree to the terms of the Contributors License
Agreement, version 1.0, with MD5 checksum
239511D9C5CD34470263BED258E9B75EDFE453C4B8F33379A8B29606787F5739

I furthermore declare that I am authorized and able to make this
agreement and sign this declaration.

Signed,

[your name]
https://github.com/[your github userid]
```

Replace the bracketed text as follows:

* `[date]` with today's date, in the unambiguous numeric form `YYYY-MM-DD`.
* `[your name]` with your name.
* `[your github userid]` with your GitHub userid.

You can confirm the MD5 checksum of the CLA by running the md5 program over `cla-1.0.md`:

```
md5 cla-1.0.md
MD5 (cla-1.0.md) =  239511D9C5CD34470263BED258E9B75EDFE453C4B8F33379A8B29606787F5739
```

If the output is different from above, do not sign the CLA and let us know.

That's it!

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Sending the Email
-----------------

Send an email to InSPIRE's official Open Sourceror
at [silvana.ovaitt@nrel.gov](mailto:silvana.ayala@nrel.gov), 
cc-ing [Jordan.Macknick@nrel.gov](mailto:Jordan.Macknick@nrel.gov), with
the subject "CLA InSPIRE" 
and the following body:

```
I submitted a pull request to indicate agreement to the terms
of the Contributors License Agreement.

Signed,

[your name]
https://github.com/[your github userid]
```

Replace the bracketed text as follows:

* `[your name]` with your name.
* `[your github userid]` with your GitHub userid.

