+++
Categories = [ "Python", "Data Engineering" ]
Description = ""
Tags = ["data_eng"]
date = "2014-03-05T17:54:38+08:00"
title = "A simple script to automate MySQLdump backups"
+++

I just moved my MySQL database to some OpenVZ VPS, which doesn't support snapshot backups. Therefore I had to set up some backup mechanism myself.

The solution I came up with is to use [BitTorrent Sync](http://www.bittorrent.com/sync) to sync my backups to the other server. It turns out to be much faster than transfering backups using scp and much easier (and perhaps more secure) than using FTP. I highly recommend BitTorrent Sync.

However, dumping mysql databases, archiving the results, and then moving into the syncing directory can still be tedious. Since the databases are big, sitting in front of the computer and waiting for the dump to complete is not an option. As a result, I wrote a simple script to automate the whole process using this amazing library [sh](https://pypi.python.org/pypi/sh). Here I provide a bootstrap code for anyone who is interested to start with:

Prerequisites
---------
+ sh (python)
+ p7zip

Bootstrap Code
---------
```python
import sh
from datetime import date

filename = "database_name." + date.today().isoformat() + ".sql"
#perform full dump on the selected database
p = sh.mysqldump("database_name", "-p'password'", u="username", r=filename, _bg=True)
p.wait()

sh.p7zip(filename)
sh.mv(filename+".7z", "backup/")
```
