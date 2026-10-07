"""The edit decision list is the single timing authority for this experiment."""
import argparse
import json
import math
from pathlib import Path

FPS = 30
STORIES = [
    ('OPENING', 'The frontier, in focus.', ''),
    ('01 / COMPUTE', 'Capital meets compute.', 'Bloomberg · 07 OCT 2026'),
    ('02 / PLATFORMS', 'Who gets the margin?', 'Financial Times via Techmeme · 07 OCT 2026'),
    ('03 / SECURITY', 'The cost of disruption.', 'Financial Times · research by Graeme Downie'),
    ('04 / MATHEMATICS', 'Output outruns review.', 'QbitAI · reporting OpenAI claims'),
    ('05 / WATERMARKS', 'A signal inside the text.', 'Machine Heart · textGrain report'),
    ('06 / GAMES', 'A decade in the making.', 'BBC · interview with Jonathan Blow'),
    ('07 / ECONOMY', 'The upside. The exposure.', 'CNBC · remarks by IMF chief Kristalina Georgieva'),
    ('08 / HEALTHCARE', 'One place for care.', 'a16z · investor announcement · 06 OCT 2026'),
    ('CLOSING', 'See you at the next signal.', ''),
]

# Absolute seconds follow the frozen pacing map. Dates, figures and qualifications
# come from its reviewed script. No independent new news claims are introduced.
# time, story, layout, headline, subline, clip, source trim, data
EDITS = [
    (0, 0, 'cover', 'ByteFront\nEspresso', 'WEDNESDAY / 07 OCTOBER 2026', None, 0, {}),
    (7.3, 0, 'agenda', 'Today, three signals.', 'Capital. Credibility. Consequences.', None, 0, {'items':[['01', 'CHIPS', '$40B in reported talks'], ['04', 'MATHEMATICS', 'Claims meet scrutiny'], ['07', 'ECONOMY', 'AI growth, with conditions']]}),
    (13.92, 1, 'hero', 'SpaceX.\nA bigger chip bet.', 'Financing talks, according to Bloomberg', 'clip-01', 0, {'tag':'ARCHIVE · SPACEX LAUNCH'}),
    (20.5, 1, 'stat', '$40B', 'Reported target to buy Nvidia chips', None, 0, {'small':'IN TALKS', 'tail':'Banks + investors → financing → chips'}),
    (25.8, 1, 'footage', 'The hardware behind the number.', 'Nvidia assembly footage · background context', 'clip-09', 4.1, {'tag':'NVIDIA · GB200 ASSEMBLY'}),
    (33.3, 1, 'cards', 'What remains open.', 'Unnamed sources. No SpaceX announcement cited.', None, 0, {'items':[['01','SIZE','Final amount'],['02','STRUCTURE','Terms of the deal'],['03','TIMING','When it could happen']]}),
    (43.42, 2, 'hero', 'Drivers want\na better deal.', 'GoTo + Grab / Southeast Asia', 'clip-07', 0, {'tag':'JAKARTA · GRABBIKE FOOTAGE'}),
    (55.8, 2, 'comparison', 'Two falls. Different windows.', 'Share-price declines reported in the FT summary', None, 0, {'items':[['GoTo','−90%','From its 2022 peak',90],['Grab','−50%','Over the past year',50]], 'delays':[1.0,6.0]}),
    (64.5, 2, 'balance', 'The pressure is on both sides.', 'The FT describes squeezed margins and worker demands.', None, 0, {'items':[['PLATFORMS','Squeezed margins'],['DRIVERS','Better deals']]}),
    (70.3, 2, 'note', 'The terms are still unclear.', 'Driver demands and company concessions are not detailed in the available summary.', None, 0, {'small':'REPORTING LIMIT'}),
    (78.84, 3, 'hero', 'A bill for\nhybrid warfare.', 'Estimated impact on the United Kingdom', 'clip-08', 0, {'tag':'WESTMINSTER · ARCHIVE', 'contain':True}),
    (85, 3, 'stat', '£2.5B', 'Up to this amount per year', None, 0, {'small':'REPORTED ESTIMATE', 'tail':'Research by Labour MP Graeme Downie'}),
    (90.5, 3, 'flow', 'A measurement problem.', 'The research highlights a missing consistent framework.', None, 0, {'items':[['ACTIVITY','What happened?'],['IMPACT','What did it cost?'],['FRAMEWORK','How is it counted?']]}),
    (98.5, 3, 'note', 'The breakdown is missing.', 'The available headline and summary do not show how the estimate was built.', None, 0, {'small':'METHOD NOT DETAILED'}),
    (106.82, 4, 'hero', 'A flood\nof papers.', 'OpenAI mathematics claims / reported by QbitAI', 'clip-12', 0, {'tag':'ILLUSTRATIVE PRINTING FOOTAGE'}),
    (112.6, 4, 'counts', 'Scale is the headline.', 'Released manuscripts and grouped results', None, 0, {'items':[['722','MANUSCRIPTS'],['372','SETS OF RESULTS']]}),
    (121, 4, 'footage', 'From number theory to geometry.', 'Riemann zeta · Hodge · BSD / claimed progress', 'clip-04', 0, {'tag':'ILLUSTRATIVE MATHEMATICS FOOTAGE'}),
    (126.3, 4, 'zeta', 'A claimed zero-free region.', 'OpenAI claim as reported by QbitAI · schematic real-part axis', None, 0, {}),
    (137.8, 4, 'stat', '~3 hours', 'ChatGPT Pro thinking-equivalent compute per result', None, 0, {'small':'OPENAI-REPORTED AVERAGE', 'tail':'A compute comparison, not a runtime guarantee'}),
    (144.3, 4, 'cards', 'Now comes the checking.', 'Verification progress differs across the repository.', None, 0, {'items':[['01','VERIFY','Check each result'],['02','FORMALIZE','Some lack Lean proofs'],['03','REVIEW','Claims await scrutiny']]}),
    (152.5, 4, 'advisors', 'Advice is not endorsement.', 'The advisory group does not endorse all results or the process.', None, 0, {'items':[['Timothy','Gowers'],['Martin','Hairer'],['Edward','Witten']]}),
    (166, 4, 'quote', 'Can people keep up?', 'Francesco Maggi worries that results are arriving faster than they can be read.', None, 0, {'small':'THE REVIEW BOTTLENECK'}),
    (176.28, 5, 'hero', 'A watermark\nyou cannot see.', 'textGrain / OpenAI text watermarking', 'clip-11', 0, {'tag':'CHATGPT INTERFACE · CONTEXT'}),
    (183, 5, 'tokens', 'textGrain', 'Entropy-calibrated watermarking', None, 0, {'small':'CONCEPTUAL SIGNAL · NOT AN ACTUAL DETECTOR'}),
    (192, 5, 'person', 'Weijie Su', 'Corresponding author / textGrain technical report', None, 0, {'small':'COPSS PRESIDENTS’ AWARD / 2026', 'tail':'Research on watermark detection → academic leave at OpenAI'}),
    (204.8, 5, 'rollout', 'Two routes into the rollout.', 'The source describes EU deployment and an API opt-in.', None, 0, {'items':[['EU','Eligible ChatGPT + Codex users','In the coming weeks'],['GLOBAL','API customers','Opt-in']]}),
    (218.4, 5, 'note', 'Detection has an access gate.', 'Open only to vetted researchers and professional institutions.', None, 0, {'small':'DETECTOR ACCESS'}),
    (226, 5, 'quote', 'No significant difference.', 'OpenAI reports this result on Astra benchmarks with watermarking enabled.', None, 0, {'small':'COMPANY-REPORTED BENCHMARK'}),
    (232, 5, 'balance', 'Make the trade-off measurable.', 'Machine Heart’s interpretation of the technical report', None, 0, {'items':[['GENERATION','Freedom to choose text'],['DETECTION','A recoverable signal']]}),
    (243.9, 6, 'hero', 'Ten years.\n~1,500 puzzles.', 'Jonathan Blow / The Order of the Sinking Star', 'clip-10', 0, {'tag':'GAME FOOTAGE · OFFICIAL LIVESTREAM'}),
    (255, 6, 'counts', 'A very long playthrough.', 'Developer estimates, as reported by the BBC', None, 0, {'items':[['~1,500','PUZZLES'],['~500h','TO COMPLETE']]}),
    (261.5, 6, 'worlds', 'Four worlds, combined.', 'Mechanics from four other puzzle games, used with permission.', None, 0, {}),
    (271.8, 6, 'stat', 'Jai', 'A programming language built from scratch', None, 0, {'small':'THE TOOLCHAIN', 'tail':'Blow says there is no AI in the production pipeline'}),
    (278, 6, 'quote', 'Where is the promised output?', 'Blow questions claims that AI makes teams ten times as productive.', None, 0, {'small':'THE DEVELOPER’S CHALLENGE'}),
    (287, 6, 'footage', '“Nightmare mode.”', 'Blow’s description of the decade-long build', 'clip-10', 0, {'tag':'GAME FOOTAGE · OFFICIAL LIVESTREAM'}),
    (293.64, 7, 'hero', 'AI changes\nthe growth equation.', 'IMF chief Kristalina Georgieva / Singapore', 'clip-05', 0, {'tag':'SINGAPORE SKYLINE · CONTEXT'}),
    (305.5, 7, 'cards', 'Growth meets constraints.', 'Georgieva calls the growth outlook underwhelming.', None, 0, {'items':[['AI','Fast advancement','Economic change'],['ENERGY','Rising costs','Price pressure'],['DEBT','Record levels','Limited room']]}),
    (315, 7, 'growth', '+0.5 pp', 'Potential addition to annual global growth, if AI is done right', None, 0, {'small':'IMF SCENARIO · NOT A FORECAST'}),
    (327, 7, 'flow', 'How disappointment could spread.', 'The warning is conditional on earnings falling short.', None, 0, {'items':[['EARNINGS','Disappoint'],['LEVERAGE','Magnifies stress'],['EXPOSURE','Spreads the shock']]}),
    (339, 7, 'stat', '>100%', 'Global public debt / GDP', None, 0, {'small':'HEADING PAST THIS LEVEL', 'tail':'Georgieva suggests a prudently hawkish policy bias'}),
    (351.42, 8, 'stat', '$120M', 'General Medicine / Series B', None, 0, {'small':'A16Z SAYS IT IS LEADING', 'tail':'The PillPack team reunites around a healthcare store'}),
    (362, 8, 'counts', 'Care, organized as a catalog.', 'Company proposition described by its investor', None, 0, {'items':[['2,900+','PRODUCTS + SERVICES'],['ONE','CONNECTED EXPERIENCE']]}),
    (369.7, 8, 'hero', 'Options.\nPrices.\nAccess.', 'Medications · labs · specialist visits', 'clip-06', 0, {'tag':'SHELVING · ILLUSTRATIVE FOOTAGE'}),
    (375.8, 8, 'flow', 'A patient journey in one place.', 'The investor’s example connects history, next steps and costs.', None, 0, {'items':[['HISTORY','Prior records'],['NEXT STEPS','Imaging + specialist'],['COST','Visible together']]}),
    (385.2, 8, 'note', 'Remember who is speaking.', 'This is the investor explaining its own investment.', None, 0, {'small':'THE SOURCE’S PERSPECTIVE'}),
    (391.68, 9, 'cover', 'The next\nsignal awaits.', 'BYTEFRONT ESPRESSO / SEE YOU TOMORROW', None, 0, {}),
    (398, 9, 'credits', 'Today’s source desk.', '07 OCTOBER 2026 / BYTEFRONT ESPRESSO', None, 0, {'items':[['Bloomberg','Financial Times / Techmeme'],['QbitAI','Machine Heart'],['BBC','CNBC / IMF remarks'],['a16z','Investor announcement']]}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    sb = json.loads((out / 'inputs/storyboard.json').read_text())
    end = math.ceil(sb['total_duration'] * FPS)
    shots = []
    events = []
    for i, row in enumerate(EDITS):
        seconds, story, layout, headline, subline, clip, trim, data = row
        start = round(seconds * FPS)
        stop = round(EDITS[i + 1][0] * FPS) if i + 1 < len(EDITS) else end
        assert stop > start
        tag, topic, source = STORIES[story]
        shot = dict(id=f'shot-{i+1:02d}', start=start, end=stop, story=story,
                    layout=layout, headline=headline, subline=subline, clip=clip,
                    source_start=trim, data=data, tag=tag, topic=topic, source=source,
                    layer_events=[0, 12, 27, 48])
        if layout == 'comparison':
            shot['layer_events'] = [0, 12, 30, 180]
        shots.append(shot)
        for layer, offset in enumerate(shot['layer_events']):
            events.append({'id':shot['id']+f'-layer-{layer}-in', 'frame':start+offset})
    timeline = {'fps':FPS, 'duration_frames':end, 'width':1920, 'height':1080,
                'audio':'inputs/audio/program_mix.wav', 'shots':shots, 'events':events,
                'source_task':'20261007-070828-b0349f',
                'audio_policy':'unchanged source program mix; final frame adds less than 1 frame',
                'convention':'zero-based half-open intervals'}
    (out / 'timeline.json').write_text(json.dumps(timeline,ensure_ascii=False,indent=2))
    rows=['# 分镜与配音共享时间线', '', '所有区间以 `timeline.json` 的整数帧为准。每镜的主信息见 headline；画面建立后再让次级信息入场。', '', '|镜头|秒|形式|画面主任务|素材|', '|---|---|---|---|---|']
    for s in shots:
        title=s['headline'].replace('\n',' / ')
        rows.append(f"|{s['id']}|{s['start']/FPS:.3f}–{s['end']/FPS:.3f}|{s['layout']}|{title}|{s['clip'] or '程序图形'}|")
    (out/'shot-plan-zh.md').write_text('\n'.join(rows)+'\n')
    print(f'{len(shots)} shots, {end} frames, {end/FPS:.6f} seconds')


if __name__ == '__main__':
    main()
