from fabric.api import local

def git_commit(branch='develop', message='update'):
	local('git add bin')
	local('git add conf')
	local('git commit -m %s' % message)
	local('git push origin %s' % branch)
