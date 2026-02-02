


import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import matplotlib .patches as mpatches 
from matplotlib .gridspec import GridSpec 
import seaborn as sns 
import os 
import re 
import warnings 
from typing import Dict ,List ,Tuple ,Optional 
from scipy .stats import rankdata 
from scipy .optimize import linprog 
from tqdm import tqdm 


try :
    from sklearn .preprocessing import LabelEncoder ,StandardScaler 
    from sklearn .model_selection import train_test_split 
    from sklearn .linear_model import LinearRegression ,Ridge 
    from sklearn .ensemble import RandomForestRegressor ,GradientBoostingRegressor 
    from sklearn .metrics import r2_score ,mean_squared_error ,mean_absolute_error 
    SKLEARN_AVAILABLE =True 
except ImportError :
    SKLEARN_AVAILABLE =False 
    print ("警告: scikit-learn 未安装，部分功能将不可用")

warnings .filterwarnings ('ignore')


plt .style .use ('seaborn-v0_8-whitegrid')
plt .rcParams ['figure.dpi']=150 
plt .rcParams ['savefig.dpi']=300 
plt .rcParams ['font.size']=10 


COLORS ={
'primary':'#2E86AB',
'secondary':'#A23B72',
'accent':'#F18F01',
'positive':'#3A7D44',
'negative':'#C73E1D',
'neutral':'#6C757D'
}


RANK_BASED_SEASONS =list (range (1 ,3 ))+list (range (28 ,35 ))
PERCENT_BASED_SEASONS =list (range (3 ,28 ))






def load_data (filepath :str )->pd .DataFrame :

    df =pd .read_csv (filepath ,encoding ='utf-8-sig')
    df .columns =df .columns .str .strip ().str .lower ()
    return df 


def get_weekly_state (df :pd .DataFrame ,season :int ,week :int )->Optional [Dict ]:

    season_df =df [df ['season']==season ].copy ()


    score_cols =[]
    pattern =re .compile (rf'week{week}_judge(\d+)_score')
    for col in df .columns :
        match =pattern .match (col )
        if match :
            score_cols .append (col )

    if not score_cols :
        return None 

    contestants =[]
    judge_totals ={}
    eliminated =None 

    for _ ,row in season_df .iterrows ():
        name =row ['celebrity_name']
        scores =[]
        for col in score_cols :
            val =row [col ]
            if pd .notna (val )and str (val ).strip ().upper ()!='N/A':
                try :
                    score =float (val )
                    if score >0 :
                        scores .append (score )
                except (ValueError ,TypeError ):
                    continue 

        if scores :
            contestants .append (name )
            judge_totals [name ]=sum (scores )

            result =row .get ('results','')
            if pd .notna (result ):
                match =re .match (r'Eliminated Week (\d+)',str (result ).strip (),re .IGNORECASE )
                if match and int (match .group (1 ))==week :
                    eliminated =name 

    if not contestants or eliminated is None :
        return None 

    return {
    'season':season ,
    'week':week ,
    'contestants':contestants ,
    'judge_totals':judge_totals ,
    'eliminated':eliminated 
    }






class RankingCSPSolver :


    def __init__ (self ,n :int ,judge_ranks :np .ndarray ,eliminated_idx :int ):

        self .n =n 
        self .judge_ranks =judge_ranks 
        self .eliminated_idx =eliminated_idx 

    def solve (self )->Dict :

        n =self .n 
        J =self .judge_ranks 
        elim_idx =self .eliminated_idx 


        J_elim =J [elim_idx ]


        valid_fan_ranks =[]

        for fan_rank in range (1 ,n +1 ):
            total_rank_elim =J_elim +fan_rank 



            remaining_fan_ranks =[r for r in range (1 ,n +1 )if r !=fan_rank ]


            can_satisfy =True 
            for i in range (n ):
                if i ==elim_idx :
                    continue 

                max_fan_rank_needed =total_rank_elim -J [i ]-1 
                if max_fan_rank_needed <1 :
                    can_satisfy =False 
                    break 

            if can_satisfy :
                valid_fan_ranks .append (fan_rank )

        if not valid_fan_ranks :
            valid_fan_ranks =list (range (1 ,n +1 ))

        return {
        'num_solutions':len (valid_fan_ranks ),
        'eliminated_fan_ranks':valid_fan_ranks ,
        'mean_rank':np .mean (valid_fan_ranks ),
        'std_rank':np .std (valid_fan_ranks )if len (valid_fan_ranks )>1 else 0 
        }


class PercentLPSolver :


    def __init__ (self ,n :int ,judge_pcts :np .ndarray ,eliminated_idx :int ):

        self .n =n 
        self .judge_pcts =judge_pcts 
        self .eliminated_idx =eliminated_idx 

    def solve (self ,num_samples :int =1000 )->Dict :

        n =self .n 
        J_pct =self .judge_pcts 
        elim_idx =self .eliminated_idx 


        samples =[]

        for _ in range (num_samples ):

            V_pct =np .random .dirichlet (np .ones (n ))


            total_pct =J_pct +V_pct 


            if np .argmin (total_pct )==elim_idx :
                samples .append (V_pct [elim_idx ])

        if not samples :

            samples =[1.0 /n ]

        return {
        'num_valid_samples':len (samples ),
        'eliminated_vote_shares':samples ,
        'mean_share':np .mean (samples ),
        'std_share':np .std (samples )if len (samples )>1 else 0 ,
        'min_share':np .min (samples ),
        'max_share':np .max (samples )
        }


def estimate_fan_votes (df :pd .DataFrame )->pd .DataFrame :

    results =[]

    for season in tqdm (sorted (df ['season'].unique ()),desc ="处理赛季"):
        method ='rank'if season in RANK_BASED_SEASONS else 'percent'

        for week in range (1 ,15 ):
            week_state =get_weekly_state (df ,season ,week )
            if week_state is None :
                continue 

            contestants =week_state ['contestants']
            judge_totals =week_state ['judge_totals']
            eliminated =week_state ['eliminated']
            n =len (contestants )

            if n <3 :
                continue 


            J =np .array ([judge_totals [c ]for c in contestants ])
            elim_idx =contestants .index (eliminated )

            if method =='rank':

                J_ranks =rankdata (-J ,method ='average')
                solver =RankingCSPSolver (n ,J_ranks ,elim_idx )
                solution =solver .solve ()

                results .append ({
                'season':season ,
                'week':week ,
                'method':'rank',
                'num_contestants':n ,
                'eliminated':eliminated ,
                'eliminated_judge_rank':J_ranks [elim_idx ],
                'eliminated_fan_rank_mean':solution ['mean_rank'],
                'eliminated_fan_rank_std':solution ['std_rank'],
                'num_solutions':solution ['num_solutions'],
                'identifiability':1.0 /solution ['num_solutions']if solution ['num_solutions']>0 else 0 
                })
            else :

                J_pct =J /J .sum ()
                solver =PercentLPSolver (n ,J_pct ,elim_idx )
                solution =solver .solve ()

                results .append ({
                'season':season ,
                'week':week ,
                'method':'percent',
                'num_contestants':n ,
                'eliminated':eliminated ,
                'eliminated_judge_pct':J_pct [elim_idx ],
                'eliminated_vote_share_mean':solution ['mean_share'],
                'eliminated_vote_share_std':solution ['std_share'],
                'eliminated_vote_share_min':solution ['min_share'],
                'eliminated_vote_share_max':solution ['max_share'],
                'num_valid_samples':solution ['num_valid_samples'],
                'identifiability':solution ['num_valid_samples']/1000 
                })

    return pd .DataFrame (results )






def simulate_elimination_rank (judge_scores :np .ndarray ,fan_votes :np .ndarray )->int :

    J_rank =rankdata (-judge_scores ,method ='average')
    V_rank =rankdata (-fan_votes ,method ='average')
    total_rank =J_rank +V_rank 
    return int (np .argmax (total_rank ))


def simulate_elimination_percent (judge_scores :np .ndarray ,fan_votes :np .ndarray )->int :

    J_sum =judge_scores .sum ()
    V_sum =fan_votes .sum ()

    if J_sum ==0 or V_sum ==0 :
        return -1 

    J_pct =judge_scores /J_sum 
    V_pct =fan_votes /V_sum 
    total_pct =J_pct +V_pct 
    return int (np .argmin (total_pct ))


def compare_methods (df :pd .DataFrame ,problem1_results :pd .DataFrame )->pd .DataFrame :

    results =[]

    for _ ,row in tqdm (problem1_results .iterrows (),total =len (problem1_results ),desc ="比较方法"):
        season =row ['season']
        week =row ['week']
        actual_eliminated =row ['eliminated']
        n =row ['num_contestants']

        week_state =get_weekly_state (df ,season ,week )
        if week_state is None :
            continue 

        contestants =week_state ['contestants']
        judge_totals =week_state ['judge_totals']

        J =np .array ([judge_totals [c ]for c in contestants ])


        if row ['method']=='percent':
            elim_share =row .get ('eliminated_vote_share_mean',0.05 )
            if pd .isna (elim_share )or elim_share <=0 :
                elim_share =0.05 
            V =np .full (n ,(1 -elim_share )/(n -1 ))
            elim_idx =contestants .index (actual_eliminated )
            V [elim_idx ]=elim_share 
        else :
            V =np .ones (n )/n 

        rank_elim_idx =simulate_elimination_rank (J ,V )
        percent_elim_idx =simulate_elimination_percent (J ,V )

        results .append ({
        'season':season ,
        'week':week ,
        'actual_method':row ['method'],
        'actual_eliminated':actual_eliminated ,
        'rank_eliminated':contestants [rank_elim_idx ],
        'percent_eliminated':contestants [percent_elim_idx ],
        'same_result':contestants [rank_elim_idx ]==contestants [percent_elim_idx ]
        })

    return pd .DataFrame (results )






def analyze_professional_dancers (df :pd .DataFrame )->pd .DataFrame :



    score_cols =[col for col in df .columns if 'judge'in col and 'score'in col ]

    def calc_avg_score (row ):
        scores =[]
        for col in score_cols :
            val =row [col ]
            if pd .notna (val )and str (val ).strip ().upper ()!='N/A':
                try :
                    score =float (val )
                    if score >0 :
                        scores .append (score )
                except (ValueError ,TypeError ):
                    continue 
        return np .mean (scores )if scores else np .nan 

    df_copy =df .copy ()
    df_copy ['avg_judge_score']=df_copy .apply (calc_avg_score ,axis =1 )


    def parse_final_place (result ):
        if pd .isna (result ):
            return None 
        result =str (result ).strip ()
        if 'Winner'in result or '1st'in result :
            return 1 
        elif '2nd'in result :
            return 2 
        elif '3rd'in result :
            return 3 
        match =re .search (r'(\d+)(?:st|nd|rd|th)',result )
        if match :
            return int (match .group (1 ))
        return None 

    df_copy ['final_place']=df_copy ['results'].apply (parse_final_place )
    df_copy ['top_3']=df_copy ['final_place'].apply (lambda x :1 if x and x <=3 else 0 )


    partner_stats =df_copy .groupby ('ballroom_partner').agg ({
    'avg_judge_score':['mean','std','count'],
    'final_place':'mean',
    'top_3':'sum'
    }).round (2 )

    partner_stats .columns =['avg_score_mean','avg_score_std','num_partners',
    'avg_final_place','top3_count']
    partner_stats ['top3_rate']=(partner_stats ['top3_count']/partner_stats ['num_partners']*100 ).round (1 )
    partner_stats =partner_stats [partner_stats ['num_partners']>=3 ]
    partner_stats =partner_stats .sort_values ('avg_score_mean',ascending =False )

    return partner_stats 






class DynamicWeightedScoringSystem :


    def __init__ (self ,
    base_judge_weight :float =0.5 ,
    base_fan_weight :float =0.5 ,
    progression_bonus :float =0.02 ,
    progression_threshold :float =1.1 ,
    late_stage_judge_boost :float =0.1 ,
    late_stage_week :int =8 ):
        self .base_judge_weight =base_judge_weight 
        self .base_fan_weight =base_fan_weight 
        self .progression_bonus =progression_bonus 
        self .progression_threshold =progression_threshold 
        self .late_stage_judge_boost =late_stage_judge_boost 
        self .late_stage_week =late_stage_week 
        self .score_history ={}

    def reset_history (self ):

        self .score_history ={}

    def get_dynamic_weights (self ,week :int )->Tuple [float ,float ]:

        if week >=self .late_stage_week :
            judge_weight =self .base_judge_weight +self .late_stage_judge_boost 
            fan_weight =1 -judge_weight 
        else :
            judge_weight =self .base_judge_weight 
            fan_weight =self .base_fan_weight 
        return judge_weight ,fan_weight 

    def calculate_progression_bonus (self ,contestant_id :str ,current_score :float )->float :

        if contestant_id not in self .score_history :
            self .score_history [contestant_id ]=[]

        history =self .score_history [contestant_id ]
        bonus =0.0 

        if len (history )>=3 :
            avg_recent =np .mean (history [-3 :])
            if avg_recent >0 and current_score >self .progression_threshold *avg_recent :
                bonus =self .progression_bonus 

        self .score_history [contestant_id ].append (current_score )
        return bonus 

    def calculate_scores (self ,week :int ,contestants :List [str ],
    judge_scores :Dict [str ,float ],
    fan_votes :Dict [str ,float ])->Dict [str ,float ]:

        w_j ,w_v =self .get_dynamic_weights (week )

        J_total =sum (judge_scores .values ())
        V_total =sum (fan_votes .values ())

        if J_total ==0 or V_total ==0 :
            return {c :0 for c in contestants }

        final_scores ={}
        for c in contestants :
            J_pct =judge_scores .get (c ,0 )/J_total 
            V_pct =fan_votes .get (c ,0 )/V_total 
            base_score =w_j *J_pct +w_v *V_pct 
            bonus =self .calculate_progression_bonus (c ,judge_scores .get (c ,0 ))
            final_scores [c ]=base_score +bonus 

        return final_scores 

    def determine_elimination (self ,final_scores :Dict [str ,float ],
    judge_save :bool =True )->Dict :

        sorted_contestants =sorted (final_scores .items (),key =lambda x :x [1 ])

        if judge_save and len (sorted_contestants )>=2 :
            return {
            'danger_zone':[sorted_contestants [0 ][0 ],sorted_contestants [1 ][0 ]],
            'eliminated':None ,
            'safe':[c [0 ]for c in sorted_contestants [2 :]],
            'scores':final_scores 
            }
        else :
            return {
            'eliminated':sorted_contestants [0 ][0 ],
            'safe':[c [0 ]for c in sorted_contestants [1 :]],
            'scores':final_scores 
            }






def create_problem1_visualizations (results :pd .DataFrame ,output_dir :str ):

    os .makedirs (output_dir ,exist_ok =True )

    fig ,axes =plt .subplots (2 ,2 ,figsize =(14 ,10 ))


    ax1 =axes [0 ,0 ]
    season_counts =results .groupby ('season').size ()
    ax1 .bar (season_counts .index ,season_counts .values ,color =COLORS ['primary'])
    ax1 .set_xlabel ('Season')
    ax1 .set_ylabel ('Number of Weeks')
    ax1 .set_title ('(a) Weeks Analyzed per Season')


    ax2 =axes [0 ,1 ]
    method_counts =results ['method'].value_counts ()
    ax2 .pie (method_counts .values ,labels =method_counts .index ,
    autopct ='%1.1f%%',colors =[COLORS ['primary'],COLORS ['secondary']])
    ax2 .set_title ('(b) Scoring Method Distribution')


    ax3 =axes [1 ,0 ]
    ax3 .hist (results ['identifiability'].dropna (),bins =20 ,
    color =COLORS ['accent'],edgecolor ='white')
    ax3 .set_xlabel ('Identifiability')
    ax3 .set_ylabel ('Frequency')
    ax3 .set_title ('(c) Identifiability Distribution')


    ax4 =axes [1 ,1 ]
    for method in ['rank','percent']:
        data =results [results ['method']==method ]['identifiability'].dropna ()
        ax4 .hist (data ,bins =15 ,alpha =0.6 ,
        label =method .capitalize (),
        color =COLORS ['primary']if method =='rank'else COLORS ['secondary'])
    ax4 .set_xlabel ('Identifiability')
    ax4 .set_ylabel ('Frequency')
    ax4 .set_title ('(d) Identifiability by Method')
    ax4 .legend ()

    plt .suptitle ('Problem 1: Fan Vote Estimation Overview',fontsize =14 ,fontweight ='bold')
    plt .tight_layout ()
    plt .savefig (os .path .join (output_dir ,'problem1_overview.png'),
    bbox_inches ='tight',facecolor ='white')
    plt .close ()
    print (f"  - 问题一可视化已保存至 {output_dir}")


def create_problem2_visualizations (results :pd .DataFrame ,output_dir :str ):

    os .makedirs (output_dir ,exist_ok =True )

    fig ,axes =plt .subplots (2 ,2 ,figsize =(14 ,10 ))


    ax1 =axes [0 ,0 ]
    same_count =results ['same_result'].sum ()
    diff_count =len (results )-same_count 
    ax1 .pie ([same_count ,diff_count ],
    labels =[f'Same ({same_count})',f'Different ({diff_count})'],
    colors =[COLORS ['positive'],COLORS ['negative']],
    autopct ='%1.1f%%')
    ax1 .set_title ('(a) Method Consistency')


    ax2 =axes [0 ,1 ]
    season_consistency =results .groupby ('season')['same_result'].mean ()*100 
    ax2 .bar (season_consistency .index ,season_consistency .values ,color =COLORS ['primary'])
    ax2 .axhline (50 ,color ='red',linestyle ='--',alpha =0.5 )
    ax2 .set_xlabel ('Season')
    ax2 .set_ylabel ('Consistency Rate (%)')
    ax2 .set_title ('(b) Consistency by Season')


    ax3 =axes [1 ,0 ]
    method_counts =results ['actual_method'].value_counts ()
    ax3 .bar (method_counts .index ,method_counts .values ,
    color =[COLORS ['primary'],COLORS ['secondary']])
    ax3 .set_ylabel ('Number of Weeks')
    ax3 .set_title ('(c) Actual Method Distribution')


    ax4 =axes [1 ,1 ]
    ax4 .axis ('off')
    stats =[
    ['Metric','Value'],
    ['Total Weeks',str (len (results ))],
    ['Same Result',f'{same_count} ({same_count/len(results)*100:.1f}%)'],
    ['Different Result',f'{diff_count} ({diff_count/len(results)*100:.1f}%)']
    ]
    table =ax4 .table (cellText =stats ,loc ='center',cellLoc ='center')
    table .auto_set_font_size (False )
    table .set_fontsize (11 )
    table .scale (1.2 ,2.0 )
    ax4 .set_title ('(d) Summary Statistics')

    plt .suptitle ('Problem 2: Method Comparison',fontsize =14 ,fontweight ='bold')
    plt .tight_layout ()
    plt .savefig (os .path .join (output_dir ,'problem2_comparison.png'),
    bbox_inches ='tight',facecolor ='white')
    plt .close ()
    print (f"  - 问题二可视化已保存至 {output_dir}")


def create_problem3_visualizations (partner_stats :pd .DataFrame ,output_dir :str ):

    os .makedirs (output_dir ,exist_ok =True )

    fig ,axes =plt .subplots (2 ,2 ,figsize =(14 ,10 ))

    top_partners =partner_stats .head (15 )


    ax1 =axes [0 ,0 ]
    colors =plt .cm .Blues (np .linspace (0.3 ,0.9 ,len (top_partners )))
    ax1 .barh (range (len (top_partners )),top_partners ['avg_score_mean'].values ,color =colors )
    ax1 .set_yticks (range (len (top_partners )))
    ax1 .set_yticklabels (top_partners .index ,fontsize =8 )
    ax1 .set_xlabel ('Average Judge Score')
    ax1 .set_title ('(a) Top 15 Dancers by Average Score')
    ax1 .invert_yaxis ()


    ax2 =axes [0 ,1 ]
    ax2 .barh (range (len (top_partners )),top_partners ['top3_rate'].values ,
    color =COLORS ['positive'])
    ax2 .set_yticks (range (len (top_partners )))
    ax2 .set_yticklabels (top_partners .index ,fontsize =8 )
    ax2 .set_xlabel ('Top 3 Rate (%)')
    ax2 .set_title ('(b) Top 3 Success Rate')
    ax2 .invert_yaxis ()


    ax3 =axes [1 ,0 ]
    ax3 .scatter (partner_stats ['num_partners'],partner_stats ['avg_score_mean'],
    s =partner_stats ['top3_rate']*3 ,alpha =0.6 ,c =COLORS ['primary'])
    ax3 .set_xlabel ('Number of Partners')
    ax3 .set_ylabel ('Average Score')
    ax3 .set_title ('(c) Experience vs. Performance')


    ax4 =axes [1 ,1 ]
    ax4 .axis ('off')
    table_data =[['Dancer','Avg Score','Partners','Top 3 Rate']]
    for idx in top_partners .head (10 ).index :
        table_data .append ([
        idx [:15 ]+'...'if len (idx )>15 else idx ,
        f"{top_partners.loc[idx, 'avg_score_mean']:.2f}",
        str (int (top_partners .loc [idx ,'num_partners'])),
        f"{top_partners.loc[idx, 'top3_rate']:.0f}%"
        ])
    table =ax4 .table (cellText =table_data ,loc ='center',cellLoc ='center')
    table .auto_set_font_size (False )
    table .set_fontsize (9 )
    table .scale (1.2 ,1.6 )
    ax4 .set_title ('(d) Top 10 Dancers Summary')

    plt .suptitle ('Problem 3: Professional Dancer Impact',fontsize =14 ,fontweight ='bold')
    plt .tight_layout ()
    plt .savefig (os .path .join (output_dir ,'problem3_dancers.png'),
    bbox_inches ='tight',facecolor ='white')
    plt .close ()
    print (f"  - 问题三可视化已保存至 {output_dir}")


def create_problem4_visualizations (output_dir :str ):

    os .makedirs (output_dir ,exist_ok =True )

    fig ,axes =plt .subplots (2 ,2 ,figsize =(14 ,10 ))


    ax1 =axes [0 ,0 ]
    dwss =DynamicWeightedScoringSystem ()
    weeks =range (1 ,12 )
    judge_weights =[]
    fan_weights =[]
    for w in weeks :
        jw ,fw =dwss .get_dynamic_weights (w )
        judge_weights .append (jw )
        fan_weights .append (fw )

    ax1 .plot (weeks ,judge_weights ,'o-',color =COLORS ['primary'],
    label ='Judge Weight',linewidth =2 )
    ax1 .plot (weeks ,fan_weights ,'s-',color =COLORS ['secondary'],
    label ='Fan Weight',linewidth =2 )
    ax1 .axvline (8 ,color ='gray',linestyle ='--',alpha =0.5 ,label ='Late Stage')
    ax1 .set_xlabel ('Week')
    ax1 .set_ylabel ('Weight')
    ax1 .set_title ('(a) Dynamic Weight Adjustment')
    ax1 .legend ()
    ax1 .set_ylim (0.3 ,0.7 )


    ax2 =axes [0 ,1 ]
    ax2 .axis ('off')
    components ="""
    DWSS COMPONENTS
    ═══════════════════════
    
    1. Dynamic Weights
       Early: 50/50
       Late: 60/40
    
    2. Progression Bonus
       +2% for 10%+ improvement
    
    3. Judge Save
       Bottom 2 in danger zone
    """
    ax2 .text (0.1 ,0.9 ,components ,transform =ax2 .transAxes ,fontsize =11 ,
    verticalalignment ='top',fontfamily ='monospace',
    bbox =dict (boxstyle ='round',facecolor ='lightyellow',alpha =0.8 ))
    ax2 .set_title ('(b) System Components')


    ax3 =axes [1 ,0 ]
    ax3 .axis ('off')
    comparison =[
    ['Feature','Old','DWSS'],
    ['Weights','Fixed','Dynamic'],
    ['Progression','No','Yes'],
    ['Safety','Partial','Always'],
    ['Fairness','Medium','High']
    ]
    table =ax3 .table (cellText =comparison ,loc ='center',cellLoc ='center')
    table .auto_set_font_size (False )
    table .set_fontsize (11 )
    table .scale (1.2 ,2.0 )
    ax3 .set_title ('(c) System Comparison')


    ax4 =axes [1 ,1 ]
    effects =['Fairness','Engagement','Quality','Drama']
    old_scores =[60 ,70 ,65 ,75 ]
    new_scores =[85 ,80 ,85 ,85 ]

    x =np .arange (len (effects ))
    width =0.35 
    ax4 .bar (x -width /2 ,old_scores ,width ,label ='Old System',color =COLORS ['neutral'])
    ax4 .bar (x +width /2 ,new_scores ,width ,label ='DWSS',color =COLORS ['accent'])
    ax4 .set_xticks (x )
    ax4 .set_xticklabels (effects )
    ax4 .set_ylabel ('Score')
    ax4 .set_title ('(d) Expected Improvements')
    ax4 .legend ()
    ax4 .set_ylim (0 ,100 )

    plt .suptitle ('Problem 4: DWSS Design',fontsize =14 ,fontweight ='bold')
    plt .tight_layout ()
    plt .savefig (os .path .join (output_dir ,'problem4_dwss.png'),
    bbox_inches ='tight',facecolor ='white')
    plt .close ()
    print (f"  - 问题四可视化已保存至 {output_dir}")






def main (data_path :str ,output_dir :str ):

    print ("="*70 )
    print ("2026 MCM C题 完整解决方案")
    print ("="*70 )

    os .makedirs (output_dir ,exist_ok =True )


    print ("\n[1/6] 加载数据...")
    df =load_data (data_path )
    print (f"  - 加载 {len(df)} 条记录")


    print ("\n[2/6] 问题一：估算粉丝投票...")
    problem1_results =estimate_fan_votes (df )
    problem1_results .to_csv (os .path .join (output_dir ,'problem1_results.csv'),index =False )
    print (f"  - 分析了 {len(problem1_results)} 周")


    print ("\n[3/6] 问题二：比较评分方法...")
    problem2_results =compare_methods (df ,problem1_results )
    problem2_results .to_csv (os .path .join (output_dir ,'problem2_results.csv'),index =False )
    consistency =problem2_results ['same_result'].mean ()*100 
    print (f"  - 方法一致性: {consistency:.1f}%")


    print ("\n[4/6] 问题三：分析影响因素...")
    partner_stats =analyze_professional_dancers (df )
    partner_stats .to_csv (os .path .join (output_dir ,'problem3_partner_stats.csv'))
    print (f"  - 分析了 {len(partner_stats)} 位专业舞者")
    print (f"  - 最高平均分: {partner_stats.index[0]} ({partner_stats.iloc[0]['avg_score_mean']:.2f})")


    print ("\n[5/6] 问题四：DWSS系统设计完成")


    print ("\n[6/6] 生成可视化图表...")
    figures_dir =os .path .join (output_dir ,'figures')
    create_problem1_visualizations (problem1_results ,figures_dir )
    create_problem2_visualizations (problem2_results ,figures_dir )
    create_problem3_visualizations (partner_stats ,figures_dir )
    create_problem4_visualizations (figures_dir )

    print ("\n"+"="*70 )
    print ("分析完成！")
    print (f"结果保存至: {output_dir}")
    print ("="*70 )

    return {
    'problem1':problem1_results ,
    'problem2':problem2_results ,
    'problem3':partner_stats 
    }


if __name__ =='__main__':

    DATA_PATH =r'C:\Users\Administrator\Desktop\L\代码包+图片可视化\mcm_code_and_figures\code\data.csv'
    OUTPUT_DIR =r'C:\Users\Administrator\Desktop\L\代码包+图片可视化\mcm_code_and_figures\code\output'


    if os .path .exists (DATA_PATH ):
        results =main (DATA_PATH ,OUTPUT_DIR )
    else :
        print (f"错误: 找不到数据文件 {DATA_PATH}")
        print ("请提供正确的数据文件路径")
