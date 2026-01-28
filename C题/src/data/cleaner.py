"""
Data Cleaner Module
Handles data validation, integrity checks, and cleaning
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    issue_type: str
    column: Optional[str]
    description: str
    severity: str  # 'error', 'warning', 'info'
    affected_rows: int = 0


@dataclass
class ValidationReport:
    """Complete validation report for a dataset"""
    is_valid: bool
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue] = field(default_factory=list)
    missing_values: Dict[str, int] = field(default_factory=dict)
    data_types: Dict[str, str] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == 'error':
            self.is_valid = False


class DataCleaner:
    """
    Validates and cleans Olympic data.
    
    Handles:
    - Data integrity validation
    - Missing value detection
    - Data type validation
    - Value range validation
    """
    
    # Expected columns for each dataset type
    MEDAL_COLUMNS = ['year', 'country', 'gold', 'silver', 'bronze', 'total']
    HOST_COLUMNS = ['year', 'host_country', 'host_city']
    PROGRAM_COLUMNS = ['year', 'sport', 'events']
    
    def __init__(self):
        """Initialize DataCleaner"""
        pass
    
    def validate_data_integrity(self, df: pd.DataFrame, 
                                 required_columns: List[str] = None,
                                 dataset_type: str = 'generic') -> ValidationReport:
        """
        Validate data integrity of a DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            dataset_type: Type of dataset ('medals', 'hosts', 'programs', 'athletes')
        """
        report = ValidationReport(
            is_valid=True,
            total_rows=len(df),
            total_columns=len(df.columns),
            data_types={col: str(df[col].dtype) for col in df.columns}
        )
        
        # Get required columns based on dataset type
        if required_columns is None:
            required_columns = self._get_required_columns(dataset_type)

        # Check for missing required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                report.add_issue(ValidationIssue(
                    issue_type='missing_column',
                    column=None,
                    description=f"Missing required columns: {missing_cols}",
                    severity='error'
                ))
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                report.missing_values[col] = missing_count
                report.add_issue(ValidationIssue(
                    issue_type='missing_values',
                    column=col,
                    description=f"Column '{col}' has {missing_count} missing values",
                    severity='warning',
                    affected_rows=missing_count
                ))
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report.add_issue(ValidationIssue(
                issue_type='duplicates',
                column=None,
                description=f"Found {duplicates} duplicate rows",
                severity='warning',
                affected_rows=duplicates
            ))
        
        # Validate data types and ranges
        self._validate_types_and_ranges(df, dataset_type, report)
        
        return report

    def _get_required_columns(self, dataset_type: str) -> List[str]:
        """Get required columns for a dataset type"""
        column_map = {
            'medals': self.MEDAL_COLUMNS,
            'hosts': self.HOST_COLUMNS,
            'programs': self.PROGRAM_COLUMNS,
            'generic': []
        }
        return column_map.get(dataset_type, [])
    
    def _validate_types_and_ranges(self, df: pd.DataFrame, 
                                    dataset_type: str,
                                    report: ValidationReport) -> None:
        """Validate data types and value ranges"""
        if dataset_type == 'medals':
            # Year should be reasonable Olympic years
            if 'year' in df.columns:
                invalid_years = df[(df['year'] < 1896) | (df['year'] > 2028)]
                if len(invalid_years) > 0:
                    report.add_issue(ValidationIssue(
                        issue_type='invalid_range',
                        column='year',
                        description=f"Found {len(invalid_years)} rows with invalid years",
                        severity='warning',
                        affected_rows=len(invalid_years)
                    ))
            
            # Medal counts should be non-negative
            for col in ['gold', 'silver', 'bronze', 'total']:
                if col in df.columns:
                    negative = df[df[col] < 0]
                    if len(negative) > 0:
                        report.add_issue(ValidationIssue(
                            issue_type='invalid_range',
                            column=col,
                            description=f"Found {len(negative)} negative values in '{col}'",
                            severity='error',
                            affected_rows=len(negative)
                        ))

    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Detect missing values in DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary mapping column names to missing value counts
        """
        missing = {}
        for col in df.columns:
            count = df[col].isna().sum()
            if count > 0:
                missing[col] = count
        return missing
    
    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get detailed summary of missing values.
        
        Returns:
            DataFrame with missing value statistics per column
        """
        summary = []
        for col in df.columns:
            missing_count = df[col].isna().sum()
            summary.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percent': (missing_count / len(df)) * 100,
                'dtype': str(df[col].dtype)
            })
        return pd.DataFrame(summary)
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names (lowercase, remove spaces).
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame with cleaned column names
        """
        df = df.copy()
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, 
                          subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
        """
        Remove duplicate rows.
        
        Returns:
            Tuple of (cleaned DataFrame, number of duplicates removed)
        """
        original_len = len(df)
        df_clean = df.drop_duplicates(subset=subset)
        removed = original_len - len(df_clean)
        return df_clean, removed
