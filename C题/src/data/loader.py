"""
Data Loader Module
Handles loading of all Olympic data CSV files
"""
import os
import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass


class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass


@dataclass
class LoadResult:
    """Result of a data load operation"""
    success: bool
    data: Optional[pd.DataFrame]
    error_message: Optional[str] = None
    rows_loaded: int = 0
    columns_loaded: int = 0


class DataLoader:
    """
    Loads Olympic data from CSV files.
    
    Handles:
    - Medal counts data
    - Host country data
    - Programs/events data
    - Athletes data
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        if data_dir is None:
            # Default to data directory relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, 'data')
        
        self.data_dir = data_dir
        self._validate_data_dir()
    
    def _validate_data_dir(self) -> None:
        """Validate that data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_csv(self, filename: str, **kwargs) -> LoadResult:
        """
        Load a CSV file and return LoadResult.
        
        Args:
            filename: Name of CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            LoadResult with success status and data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return LoadResult(
                success=False,
                data=None,
                error_message=f"File not found: {filepath}"
            )
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                return LoadResult(
                    success=True,
                    data=df,
                    rows_loaded=len(df),
                    columns_loaded=len(df.columns)
                )
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return LoadResult(
                    success=False,
                    data=None,
                    error_message=f"Error loading {filename}: {str(e)}"
                )
        
        return LoadResult(
            success=False,
            data=None,
            error_message=f"Error loading {filename}: Could not decode with any encoding"
        )
    
    def load_medals(self, filename: str = 'summerOly_medal_counts.csv') -> pd.DataFrame:
        """
        Load medal counts data.
        
        Args:
            filename: Name of medals CSV file
            
        Returns:
            DataFrame with medal counts
            
        Raises:
            DataLoadError: If file cannot be loaded
        """
        result = self._load_csv(filename)
        
        if not result.success:
            raise DataLoadError(result.error_message)
        
        return result.data
    
    def load_hosts(self, filename: str = 'summerOly_hosts.csv') -> pd.DataFrame:
        """
        Load host country data.
        
        Args:
            filename: Name of hosts CSV file
            
        Returns:
            DataFrame with host information
            
        Raises:
            DataLoadError: If file cannot be loaded
        """
        result = self._load_csv(filename)
        
        if not result.success:
            raise DataLoadError(result.error_message)
        
        return result.data
    
    def load_programs(self, filename: str = 'summerOly_programs.csv') -> pd.DataFrame:
        """
        Load Olympic programs/events data.
        
        Args:
            filename: Name of programs CSV file
            
        Returns:
            DataFrame with program information
            
        Raises:
            DataLoadError: If file cannot be loaded
        """
        result = self._load_csv(filename)
        
        if not result.success:
            raise DataLoadError(result.error_message)
        
        return result.data
    
    def load_athletes(self, filename: str = 'summerOly_athletes.csv') -> pd.DataFrame:
        """
        Load athletes data.
        
        Args:
            filename: Name of athletes CSV file
            
        Returns:
            DataFrame with athlete information
            
        Raises:
            DataLoadError: If file cannot be loaded
        """
        result = self._load_csv(filename)
        
        if not result.success:
            raise DataLoadError(result.error_message)
        
        return result.data
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all four CSV files.
        
        Returns:
            Dictionary with keys 'medals', 'hosts', 'programs', 'athletes'
            
        Raises:
            DataLoadError: If any file cannot be loaded
        """
        data = {}
        errors = []
        
        # Try to load each file
        try:
            data['medals'] = self.load_medals()
        except DataLoadError as e:
            errors.append(str(e))
        
        try:
            data['hosts'] = self.load_hosts()
        except DataLoadError as e:
            errors.append(str(e))
        
        try:
            data['programs'] = self.load_programs()
        except DataLoadError as e:
            errors.append(str(e))
        
        try:
            data['athletes'] = self.load_athletes()
        except DataLoadError as e:
            errors.append(str(e))
        
        if errors:
            raise DataLoadError(f"Failed to load files: {'; '.join(errors)}")
        
        return data
    
    def get_load_summary(self) -> Dict[str, Any]:
        """
        Get summary of available data files.
        
        Returns:
            Dictionary with file availability and basic stats
        """
        summary = {}
        files = {
            'medals': 'summerOly_medal_counts.csv',
            'hosts': 'summerOly_hosts.csv',
            'programs': 'summerOly_programs.csv',
            'athletes': 'summerOly_athletes.csv'
        }
        
        for name, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                result = self._load_csv(filename)
                summary[name] = {
                    'exists': True,
                    'rows': result.rows_loaded if result.success else 0,
                    'columns': result.columns_loaded if result.success else 0,
                    'error': result.error_message
                }
            else:
                summary[name] = {
                    'exists': False,
                    'rows': 0,
                    'columns': 0,
                    'error': f"File not found: {filepath}"
                }
        
        return summary
