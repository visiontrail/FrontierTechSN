"""Render a frozen editorial edit with Pillow graphics and FFmpeg.

No browser automation, TTS calls, runtime API mutations, or source media writes.
All time comes from timeline.json; per-shot rendering can be resumed safely.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
PAPER = '#eeeee7'
INK = '#112c2d'
LIME = '#d5f286'
MUTED = '#82938e'
RULE = '#c8d0c5'
ORANGE = '#bc4931'
FONTS = Path('/System/Library/Fonts/Supplemental')
FONT_FILES = {'sans':FONTS/'Arial.ttf', 'bold':FONTS/'Arial Bold.ttf',
              'serif':FONTS/'Georgia.ttf', 'italic':FONTS/'Georgia Italic.ttf'}
FONT_CACHE = {}
TEXT_BOXES = []


def font(size, face='sans'):
    key=(size,face)
    if key not in FONT_CACHE:
        FONT_CACHE[key]=ImageFont.truetype(str(FONT_FILES[face]),size)
    return FONT_CACHE[key]


def canvas(color=None):
    return Image.new('RGBA',(W,H),color or (0,0,0,0))


def text(im, xy, content, size=42, fill=INK, face='sans', width=None, spacing=1.17):
    """Wrap by measured glyph widths; reject overflow instead of silently clipping."""
    d=ImageDraw.Draw(im)
    lines=[]
    for line in content.split('\n'):
        if width is None:
            lines.append(line)
            continue
        cur=''
        for word in line.split():
            nxt=(cur+' '+word).strip()
            if cur and d.textlength(nxt,font=font(size,face))>width:
                lines.append(cur);cur=word
            else: cur=nxt
        lines.append(cur)
    x,y=xy
    for line in lines:
        box=d.textbbox((x,y),line,font=font(size,face),anchor='lt')
        if box[2]>W-45 or box[3]>H-35 or box[0]<0 or box[1]<0:
            raise ValueError(f'Text outside safe canvas: {line} {box}')
        if width is not None and d.textlength(line,font=font(size,face))>width+2:
            raise ValueError(f'Word too wide: {line}')
        TEXT_BOXES.append({'text':line,'bbox':box})
        d.text((x,y),line,font=font(size,face),fill=fill,anchor='lt')
        y+=round(size*spacing)
    return y


def line(im,a,b,fill=RULE,width=2):
    ImageDraw.Draw(im).line([a,b],fill=fill,width=width)


def rect(im,box,fill,outline=None,width=1):
    ImageDraw.Draw(im).rectangle(box,fill=fill,outline=outline,width=width)


def disk(im,center,r,fill):
    x,y=center;ImageDraw.Draw(im).ellipse((x-r,y-r,x+r,y+r),fill=fill)


def arrow(im,a,b,fill=INK,width=3):
    line(im,a,b,fill,width)
    x,y=b
    ImageDraw.Draw(im).polygon([(x,y),(x-15,y-9),(x-15,y+9)],fill=fill)


def header(im,s,dark=False):
    c=PAPER if dark else INK
    text(im,(88,55),'BYTEFRONT',27,c,'bold')
    text(im,(287,56),'ESPRESSO',27,c)
    text(im,(1470,57),'07 OCT 2026',27,c)
    line(im,(88,109),(1832,109),'#45605a' if dark else RULE)
    text(im,(88,143),s['tag'],25,LIME if dark else ORANGE,'bold')
    text(im,(88,1004),s['source'] or 'BYTEFRONT ESPRESSO  /  THE DAILY FRONTIER-TECH BRIEF',24,'#b9cbc1' if dark else '#526861')
    if 1<=s['story']<=8:
        for i in range(8):
            rect(im,(1588+i*30,1009,1606+i*30,1015),LIME if dark and i<s['story'] else INK if i<s['story'] else '#698075' if dark else RULE)
        text(im,(1587,1026),f"{s['story']:02} / 08",17,'#a5b6ae' if dark else '#647970')


def big_header(layers,s,dark=False,size=84):
    color=PAPER if dark else INK
    y=text(layers[0],(88,235),s['headline'],size,color,'serif',width=1715)
    text(layers[1],(91,y+34),s['subline'],36,'#b6c8bf' if dark else '#526861',width=1690)
    return y+126


def artwork(s,clips,out):
    global TEXT_BOXES
    TEXT_BOXES=[]
    kind=s['layout'];data=s['data']
    dark=kind in ['cover','quote','tokens','person','zeta','worlds','credits']
    base=canvas(INK if dark else PAPER)
    layers=[canvas() for _ in range(4)]
    media=None
    if kind=='footage':
        base=canvas()
        # A real gradient keeps both original footage and type readable.
        for y in range(H):
            alpha=int(max(0,1-y/250)*170 + max(0,(y-410)/670)*242)
            ImageDraw.Draw(base).line([(0,y),(W,y)],fill=(8,27,29,min(240,alpha)))
        header(base,s,True)
        text(layers[0],(88,710),s['headline'],78,PAPER,'serif',width=1650)
        text(layers[1],(92,841),s['subline'],38,PAPER,width=1650)
        text(layers[2],(91,641),data['tag'],26,LIME,'bold')
        media=(0,0,W,H)
    else:
        header(base,s,dark)
    if kind=='hero':
        rect(base,(811,208,1831,845),'#dae1d6')
        text(layers[0],(88,261),s['headline'],83,INK,'serif',width=660)
        text(layers[1],(91,675),s['subline'],37,'#48655a',width=645)
        text(layers[2],(811,885),data['tag'],23,'#526861','bold',width=1020)
        media=(811,208,1020,638)
        if data.get('contain'):
            media=(995,347,640,360)
            text(layers[3],(998,745),'ARCHIVE / ORIGINAL FRAME',22,'#526861')
    elif kind=='cover':
        for x in range(1200,1860,38):line(base,(x,185),(x,886),'#24413b',1)
        disk(base,(1739,545),66,LIME)
        disk(base,(1739,545),40,INK)
        disk(base,(1530,545),215,LIME)
        disk(base,(1530,545),170,INK)
        disk(base,(1474,488),37,'#35564b')
        for i in range(3):
            line(layers[2],(1350,780+i*27),(1660-i*50,780+i*27),LIME,3)
        text(layers[0],(89,279),s['headline'],146,PAPER,'serif',width=1230)
        text(layers[1],(94,735),s['subline'],28,LIME,'bold',width=1250)
        text(layers[3],(95,840),'A concentrated shot of what comes next.',34,'#b5c9bd')
    elif kind=='agenda':
        big_header(layers,s,size=92)
        for i,item in enumerate(data['items']):
            x=90+i*589
            rect(base,(x,565,x+552,895),'#e1e6db')
            text(layers[2 if i<2 else 3],(x+30,600),item[0],66,ORANGE,'serif')
            text(layers[2 if i<2 else 3],(x+30,714),item[1],29,INK,'bold')
            text(layers[2 if i<2 else 3],(x+30,776),item[2],33,INK,width=470)
    elif kind=='stat':
        text(layers[0],(92,247),data['small'],30,ORANGE,'bold')
        size=236 if len(s['headline'])<8 else 196
        text(layers[1],(82,331),s['headline'],size,INK,'serif')
        line(base,(92,642),(1829,642),RULE,2)
        text(layers[2],(94,701),s['subline'],55,INK,width=1570)
        text(layers[3],(96,846),data['tail'],34,'#526861',width=1640)
        # Graphic ruler expresses scale without pretending to be a measured series.
        for i in range(20):line(base,(1230+i*29,309),(1230+i*29,515 if i%5==0 else 473),'#ced7c7',3)
    elif kind in ['cards','flow']:
        big_header(layers,s,size=82)
        for i,item in enumerate(data['items']):
            x=90+i*589
            rect(base,(x,563,x+549,884),'#e1e6db')
            target=layers[min(3,1+i)]
            text(target,(x+31,599),item[0],28,ORANGE,'bold')
            text(target,(x+31,667),item[1],43,INK,'serif',width=480)
            if len(item)>2:text(target,(x+31,755),item[2],33,'#526861',width=480)
            if kind=='flow' and i<2:arrow(layers[3],(x+552,721),(x+581,721),ORANGE)
    elif kind=='comparison':
        big_header(layers,s,size=85)
        for i,(name,value,window,amount) in enumerate(data['items']):
            x=95+i*890;target=layers[i+2]
            text(target,(x,515),name,47,INK,'bold')
            text(target,(x,598),value,142,ORANGE,'serif')
            rect(base,(x,801,x+720,819),'#d4dccf')
            rect(target,(x,801,x+720*amount/100,819),ORANGE)
            text(target,(x,861),window,33,INK)
        line(base,(934,500),(934,920),RULE)
    elif kind=='balance':
        big_header(layers,s,size=82)
        for i,item in enumerate(data['items']):
            x=93+i*903;target=layers[i+2]
            rect(base,(x,583,x+826,855),INK if i==0 else '#dae4cf')
            text(target,(x+42,625),item[0],29,LIME if i==0 else ORANGE,'bold')
            text(target,(x+42,709),item[1],50,PAPER if i==0 else INK,'serif',width=740)
        disk(layers[3],(945,718),38,LIME)
        text(layers[3],(927,691),'↔',45,INK)
    elif kind=='note':
        rect(base,(89,237,105,868),ORANGE)
        text(layers[0],(151,246),data['small'],29,ORANGE,'bold')
        y=text(layers[1],(144,362),s['headline'],107,INK,'serif',width=1600)
        text(layers[2],(151,y+67),s['subline'],48,'#526861',width=1500)
    elif kind=='counts':
        big_header(layers,s,size=82)
        for i,(num,label) in enumerate(data['items']):
            x=90+i*900;target=layers[i+2]
            text(target,(x,553),num,165 if len(num)<7 else 149,INK,'serif')
            line(target,(x,760),(x+770,760),ORANGE,5)
            text(target,(x+4,808),label,31,'#526861','bold',width=810)
    elif kind=='zeta':
        big_header(layers,s,True,size=87)
        y=691;x0=170;x1=1730
        line(base,(x0,y),(x1,y),'#698279',3)
        for v,label in [(0,'0'),(.5,'1/2'),(.875,'7/8'),(1,'1')]:
            x=x0+(x1-x0)*v
            line(base,(x,y-13),(x,y+13),PAPER,3)
            text(layers[2],(x-27,y+33),label,39,PAPER)
        rect(layers[1],(x0+(x1-x0)*.875,y-146,x1,y-26),LIME)
        text(layers[2],(1270,467),'CLAIM: Re(s) > 7/8',34,LIME,'bold')
        line(layers[2],(x0+(x1-x0)*.5,531),(x0+(x1-x0)*.5,657),'#e8bfa1',4)
        text(layers[3],(180,825),'RH concerns the nontrivial zeros on Re(s) = 1/2.',35,PAPER)
        text(layers[3],(180,883),'This reported claim does not settle the full hypothesis.',31,'#b5c9bd')
    elif kind=='advisors':
        big_header(layers,s,size=86)
        for i,(first,last) in enumerate(data['items']):
            x=92+i*589;target=layers[min(i+1,3)]
            line(base,(x,585),(x+528,585),ORANGE,3)
            text(target,(x,632),first,43,'#526861')
            text(target,(x,710),last,77,INK,'serif')
        text(layers[3],(95,866),'Advisory involvement is separate from verification.',34,'#526861')
    elif kind=='quote':
        text(layers[0],(94,243),data['small'],28,LIME,'bold')
        y=text(layers[1],(85,359),s['headline'],111,PAPER,'serif',width=1700)
        text(layers[2],(94,y+64),s['subline'],45,'#b5c9bd',width=1530)
        line(layers[3],(94,894),(442,894),LIME,5)
    elif kind=='tokens':
        text(layers[0],(88,232),s['headline'],144,PAPER,'serif')
        text(layers[1],(94,414),s['subline'],43,'#b5c9bd')
        for row in range(3):
            for col in range(12):
                x=96+col*142;y=570+row*82
                rect(base,(x,y,x+121,y+51),'#2e4942')
                if (row*7+col*3)%5<2:rect(layers[2],(x,y,x+121,y+51),LIME)
        text(layers[3],(96,865),data['small'],25,'#b5c9bd')
    elif kind=='person':
        text(layers[0],(94,255),'THE PERSON BEHIND THE SIGNAL',28,LIME,'bold')
        text(layers[1],(84,365),s['headline'],181,PAPER,'serif')
        text(layers[2],(94,594),s['subline'],44,'#b5c9bd')
        text(layers[2],(94,716),data['small'],34,LIME,'bold')
        text(layers[3],(94,836),data['tail'],33,'#b5c9bd',width=1700)
    elif kind=='rollout':
        big_header(layers,s,size=85)
        for i,item in enumerate(data['items']):
            x=95+i*900;target=layers[i+2]
            rect(base,(x,530,x+820,894),'#dfe6d6')
            text(target,(x+35,571),item[0],67,ORANGE,'serif')
            text(target,(x+38,687),item[1],38,INK,width=740)
            text(target,(x+38,799),item[2],32,'#526861')
    elif kind=='worlds':
        big_header(layers,s,True,size=91)
        for i in range(4):
            x=105+i*439
            rect(base,(x,580,x+387,854),'#27423b')
            for r in range(3):
                for c in range(5):
                    if (r+c+i)%3!=0:rect(layers[2],(x+29+c*66,610+r*66,x+72+c*66,653+r*66),LIME if c==i else '#729082')
            text(layers[3],(x+18,891),f'WORLD {i+1}',24,LIME,'bold')
    elif kind=='growth':
        text(layers[0],(96,242),data['small'],29,ORANGE,'bold')
        text(layers[1],(87,342),s['headline'],217,INK,'serif')
        text(layers[1],(98,645),s['subline'],43,'#526861',width=980)
        for i,(num,height) in enumerate([('3.0%',259),('3.5%',302)]):
            x=1250+i*290
            rect(base,(x,783-height,x+190,783),'#d4ddcc')
            rect(layers[2+i],(x,783-height,x+190,783),INK if i==0 else '#86a747')
            text(layers[2+i],(x-2,815),num,51,INK,'serif')
        text(layers[3],(1250,903),'Illustrative annual rates',27,'#526861')
    elif kind=='credits':
        text(layers[0],(88,230),s['headline'],104,PAPER,'serif')
        text(layers[1],(94,369),s['subline'],26,LIME,'bold')
        for i,(a,b) in enumerate(data['items']):
            y=500+i*94
            text(layers[2],(96,y),a,40,PAPER,'serif')
            text(layers[3],(856,y),b,35,'#b5c9bd',width=945)
        text(layers[3],(96,909),'Footage credits and source links accompany the review copy.',25,'#b5c9bd')
    elif kind not in ['hero','footage']:
        raise ValueError(kind)
    dest=out/'graphics'/s['id'];dest.mkdir(parents=True,exist_ok=True)
    base.save(dest/'base.png')
    for i,im in enumerate(layers):im.save(dest/f'layer-{i}.png')
    (dest/'text-boxes.json').write_text(json.dumps(TEXT_BOXES,indent=2))
    return media


def preview(s,media,clips,out):
    dest=out/'graphics'/s['id'];im=Image.open(dest/'base.png').convert('RGBA')
    if media:
        c=clips[s['clip']];sec=s['source_start']+min(2,(s['end']-s['start'])/60)
        f=dest/'footage.jpg'
        subprocess.run(['ffmpeg','-v','error','-ss',str(sec),'-i',str(out/'inputs'/c['local_path']),'-frames:v','1','-y',str(f)],check=True)
        from PIL import ImageOps
        x,y,w,h=media
        m=ImageOps.fit(Image.open(f).convert('RGBA'),(w,h))
        if s['layout']=='footage':
            bg=canvas();bg.paste(m,(x,y));bg.alpha_composite(im);im=bg
        else:im.paste(m,(x,y))
    for i in range(4):im.alpha_composite(Image.open(dest/f'layer-{i}.png'))
    f=out/'stills'/f"{s['id']}.jpg";f.parent.mkdir(exist_ok=True)
    im.convert('RGB').save(f,quality=93)
    return f


def render_shot(s,media,clips,out,fps,encoder):
    dest=out/'graphics'/s['id'];target=out/'shots'/f"{s['id']}.mp4"
    target.parent.mkdir(exist_ok=True)
    # Cache covers edit, all generated layers and the exact input snapshot.
    content=json.dumps(s,sort_keys=True).encode()+encoder.encode()+Path(__file__).read_bytes()
    for f in sorted(dest.glob('*.png')):content+=hashlib.sha256(f.read_bytes()).digest()
    content+=(out/'input-manifest.json').read_bytes()
    key=hashlib.sha256(content).hexdigest()
    stamp=target.with_suffix('.sha256')
    if target.exists() and stamp.exists() and stamp.read_text()==key:return f"{s['id']} cached"
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-filter_complex_threads','1']
    command+=['-loop','1','-framerate',str(fps),'-i',str(dest/'base.png')]
    for i in range(4):command+=['-loop','1','-framerate',str(fps),'-i',str(dest/f'layer-{i}.png')]
    duration=(s['end']-s['start'])/fps
    filters=['[0:v]format=rgba[b]'];cur='b'
    if media:
        c=clips[s['clip']];x,y,w,h=media
        available=c['duration_seconds']-s['source_start']
        if available<duration-0.04:
            # Only a short final hold is permitted; never silently loop evidence.
            if duration-available>0.8:raise ValueError(f"{s['id']}: excessive footage freeze {duration-available}")
        command+=['-ss',str(s['source_start']),'-i',str(out/'inputs'/c['local_path'])]
        filters.append(f'[5:v]setpts=PTS-STARTPTS,fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,tpad=stop_mode=clone:stop_duration=1,format=rgba[v]')
        if s['layout']=='footage':filters.append('[v][b]overlay=0:0:format=auto[m]')
        else:filters.append(f'[b][v]overlay={x}:{y}:format=auto[m]')
        cur='m'
    for i,fr in enumerate(s['layer_events']):
        delay=fr/fps
        filters.append(f'[{i+1}:v]format=rgba,fade=t=in:st={delay}:d=0.4:alpha=1[l{i}]')
        # cubic ease-out, deterministic at composition-relative frame time
        y=f"28*pow(1-min(max((t-{delay})/0.65,0),1),3)"
        filters.append(f"[{cur}][l{i}]overlay=x=0:y='{y}':format=auto[o{i}]")
        cur=f'o{i}'
    filters.append(f'[{cur}]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setsar=1[out]')
    command+=['-filter_complex',';'.join(filters),'-map','[out]','-frames:v',str(s['end']-s['start']),'-an','-r',str(fps),'-c:v',encoder]
    if encoder=='h264_videotoolbox':command+=['-b:v','12M','-maxrate','18M','-profile:v','high','-allow_sw','1']
    else:command+=['-crf','19','-preset','veryfast','-threads','2']
    command+=['-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-movflags','+faststart',str(target)]
    began=time.monotonic()
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    (target.with_suffix('.command.json')).write_text(json.dumps(command,indent=2))
    if result.returncode:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"{s['id']}: {result.stderr[-3000:]}")
    stamp.write_text(key)
    return f"{s['id']} rendered in {time.monotonic()-began:.1f}s"


def mux_final(out,timeline):
    shots=timeline['shots']
    concat=out/'shots.txt'
    concat.write_text(''.join(f"file 'shots/{s['id']}.mp4'\nduration {(s['end']-s['start'])/timeline['fps']:.9f}\n" for s in shots))
    final=out/'bytefront-director-v1.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(concat),'-i',str(out/timeline['audio']),'-map','0:v:0','-map','1:a:0','-c:v','copy','-bsf:v','h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0','-c:a','aac','-b:a','256k','-af','apad','-t',str(timeline['duration_frames']/timeline['fps']),'-video_track_timescale','90000','-movflags','+faststart',str(final)],check=True)
    print(f'DELIVERED {final}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--mux-only',action='store_true')
    parser.add_argument('--only',default='')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--encoder',default='libx264',choices=['libx264','h264_videotoolbox'])
    args=parser.parse_args();out=args.out.resolve()
    timeline=json.loads((out/'timeline.json').read_text())
    if args.mux_only:
        mux_final(out,timeline)
        return
    clips={c['id']:c for c in json.loads((out/'inputs/footage/manifest.json').read_text())['clips']}
    shots=timeline['shots']
    if args.only:shots=[s for s in shots if s['id'] in args.only.split(',')]
    media={}
    for s in shots:
        media[s['id']]=artwork(s,clips,out)
        preview(s,media[s['id']],clips,out)
    # A contact sheet is an actual visual artifact, not a proxy for motion review.
    thumbw=480;thumbh=300
    sheet=Image.new('RGB',(thumbw*4,thumbh*math.ceil(len(shots)/4)),PAPER)
    for i,s in enumerate(shots):
        x=i%4*thumbw;y=i//4*thumbh
        sheet.paste(Image.open(out/'stills'/f"{s['id']}.jpg").resize((480,270)),(x,y))
        ImageDraw.Draw(sheet).text((x+6,y+275),s['id']+' / '+s['layout'],font=font(18),fill=INK)
    sheet.save(out/'evidence'/('contact-sheet-selected.jpg' if args.only else 'contact-sheet.jpg'),quality=90)
    print(f'{len(shots)} shot graphics and previews complete',flush=True)
    if args.render:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(render_shot,s,media[s['id']],clips,out,timeline['fps'],args.encoder) for s in shots]
            for f in as_completed(futures):print(f.result(),flush=True)
    if args.render and not args.only:
        mux_final(out,timeline)


if __name__=='__main__':
    main()
