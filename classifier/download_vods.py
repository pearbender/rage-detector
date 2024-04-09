import subprocess
import json
import sys

if len(sys.argv) != 2:
	print("Usage: python script.py <TwitchAuthToken>")
	sys.exit(1)

twitch_auth_token = sys.argv[1]
vods_json = subprocess.check_output(['twitch-dl', 'videos', 'perokichi_neet', '-j', '-a']).decode('utf-8')
vods_data = json.loads(vods_json)

for vod in vods_data['videos']:
	vod_id = vod['id']
	command = [
		'twitch-dl', 'download',
		'-q', '160p',
		'--auth-token', twitch_auth_token,
		vod_id,
		'--output', f'data\\vods\\{vod_id}.mkv',
		'-w', '32'
	]
	subprocess.run(command)