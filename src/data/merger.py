"""
Data Merger Module
Handles merging of multiple Olympic datasets
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MergeResult:
    """Result of a merge operation"""
    success: bool
    data: Optional[pd.DataFrame]
    rows_before: Dict[str, int]
    rows_after: int
    unmatched_keys: Dict[str, int]
    error_message: Optional[str] = None


class DataMerger:
    """
    Merges multiple Olympic datasets on common keys.
    
    Primary merge keys: year, country
    """
    
    def __init__(self):
        """Initialize DataMerger"""
        pass
    
    def merge_datasets(self, 
                       medals: pd.DataFrame,
                       hosts: pd.DataFrame,
                       programs: pd.DataFrame = None,
                       athletes: pd.DataFrame = None) -> MergeResult:
        """
        Merge all datasets into a unified analysis dataset.
        
        Args:
            medals: Medal counts DataFrame
            hosts: Host country DataFrame
            programs: Programs/events DataFrame (optional)
            athletes: Athletes DataFrame (optional)
        """
        rows_before = {'medals': len(medals), 'hosts': len(hosts)}
        unmatched = {}

        try:
            # Start with medals as base
            merged = medals.copy()
            
            # Merge with hosts on year
            if 'year' in hosts.columns:
                # Rename host columns to avoid conflicts
                hosts_renamed = hosts.rename(columns={
                    'host_country': 'host_country',
                    'host_city': 'host_city'
                })
                merged = merged.merge(
                    hosts_renamed[['year', 'host_country', 'host_city']],
                    on='year',
                    how='left'
                )
                # Track unmatched
                unmatched['hosts'] = merged['host_country'].isna().sum()
            
            # Add host indicator
            if 'host_country' in merged.columns and 'country' in merged.columns:
                merged['is_host'] = (merged['country'] == merged['host_country']).astype(int)
            
            # Merge with programs if provided
            if programs is not None:
                rows_before['programs'] = len(programs)
                # Aggregate programs by year
                if 'year' in programs.columns and 'events' in programs.columns:
                    program_agg = programs.groupby('year').agg({
                        'events': 'sum'
                    }).reset_index()
                    program_agg.columns = ['year', 'total_events']
                    merged = merged.merge(program_agg, on='year', how='left')
                    unmatched['programs'] = merged['total_events'].isna().sum()

            # Merge with athletes if provided
            if athletes is not None:
                rows_before['athletes'] = len(athletes)
                # Aggregate athletes by year and country
                if all(col in athletes.columns for col in ['year', 'country']):
                    athlete_agg = athletes.groupby(['year', 'country']).size().reset_index()
                    athlete_agg.columns = ['year', 'country', 'athlete_count']
                    merged = merged.merge(athlete_agg, on=['year', 'country'], how='left')
                    unmatched['athletes'] = merged['athlete_count'].isna().sum()
            
            return MergeResult(
                success=True,
                data=merged,
                rows_before=rows_before,
                rows_after=len(merged),
                unmatched_keys=unmatched
            )
            
        except Exception as e:
            return MergeResult(
                success=False,
                data=None,
                rows_before=rows_before,
                rows_after=0,
                unmatched_keys=unmatched,
                error_message=str(e)
            )
    
    def merge_on_keys(self, 
                      left: pd.DataFrame, 
                      right: pd.DataFrame,
                      keys: List[str],
                      how: str = 'left') -> Tuple[pd.DataFrame, int]:
        """
        Merge two DataFrames on specified keys.
        
        Returns:
            Tuple of (merged DataFrame, count of unmatched rows)
        """
        merged = left.merge(right, on=keys, how=how)
        
        # Count unmatched (NaN in right-side columns)
        right_cols = [c for c in right.columns if c not in keys]
        if right_cols:
            unmatched = merged[right_cols[0]].isna().sum()
        else:
            unmatched = 0
        
        return merged, unmatched
    
    def validate_merge_keys(self, 
                            df1: pd.DataFrame, 
                            df2: pd.DataFrame,
                            keys: List[str]) -> Dict[str, bool]:
        """
        Validate that merge keys exist in both DataFrames.
        """
        result = {}
        for key in keys:
            result[key] = (key in df1.columns) and (key in df2.columns)
        return result
