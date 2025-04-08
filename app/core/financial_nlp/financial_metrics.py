import logging
from typing import Dict, Any, List, Union, Optional
import re
import pandas as pd
import numpy as np
from datetime import datetime
import json
from ...config.settings import settings

logger = logging.getLogger(__name__)

class FinancialMetricsAnalyzer:
    def __init__(self):
        pass
    
    def extract_financial_values(self, text: str) -> Dict[str, Any]:
        """
        Extract financial values from text
        
        Args:
            text: Financial text
            
        Returns:
            Dictionary of extracted values
        """
        values = {
            "revenues": self._extract_values(text, ["revenue", "sales"], "amount"),
            "profits": self._extract_values(text, ["profit", "income", "earnings"], "amount"),
            "margins": self._extract_values(text, ["margin"], "percentage"),
            "growth_rates": self._extract_values(text, ["growth", "increase"], "percentage")
        }
        
        return values
    
    def _extract_values(self, text: str, keywords: List[str], value_type: str) -> List[Dict[str, Any]]:
        """
        Extract specific value types from text
        
        Args:
            text: Financial text
            keywords: List of keywords to look for
            value_type: Type of value to extract (amount or percentage)
            
        Returns:
            List of extracted values with context
        """
        results = []
        
        # Create regex pattern for keywords
        keyword_pattern = '|'.join(keywords)
        
        if value_type == "amount":
            # Pattern for amounts like $1.2 billion, 1.2 million, etc.
            pattern = rf"([\w\s]{{0,30}})({keyword_pattern})([\w\s]{{0,30}})([$]?[\d,]+\.?\d*)\s?(million|billion|trillion|M|B|T)?([\w\s]{{0,30}})"
        else:  # percentage
            # Pattern for percentages like 12.3%, 12.3 percent, etc.
            pattern = rf"([\w\s]{{0,30}})({keyword_pattern})([\w\s]{{0,30}})([\d\.]+)[%\s]+(percent|pct)?([\w\s]{{0,30}})"
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            prefix = match.group(1)
            keyword = match.group(2)
            middle = match.group(3)
            value = match.group(4)
            unit = match.group(5) if match.group(5) else ""
            suffix = match.group(6)
            
            # Clean and convert value
            value = value.replace(",", "").replace("$", "")
            try:
                numeric_value = float(value)
                
                # Apply multiplier for amounts
                if value_type == "amount" and unit:
                    if unit.lower() in ["billion", "b"]:
                        numeric_value *= 1_000_000_000
                    elif unit.lower() in ["million", "m"]:
                        numeric_value *= 1_000_000
                    elif unit.lower() in ["trillion", "t"]:
                        numeric_value *= 1_000_000_000_000
                
                # Create context string
                context = f"{prefix}{keyword}{middle}{value} {unit}{suffix}"
                
                # Extract time period if available
                time_periods = self._extract_time_periods(context)
                
                result = {
                    "keyword": keyword,
                    "value": numeric_value,
                    "unit": unit,
                    "context": context.strip(),
                    "time_periods": time_periods
                }
                
                results.append(result)
                
            except ValueError:
                continue
        
        return results
    
    def _extract_time_periods(self, text: str) -> List[str]:
        """
        Extract time periods from text
        
        Args:
            text: Text to extract time periods from
            
        Returns:
            List of time periods
        """
        periods = []
        
        # Year pattern
        year_pattern = r'\b(20\d{2}|19\d{2})\b'
        years = re.findall(year_pattern, text)
        
        # Quarter pattern
        quarter_pattern = r'\b(?:Q[1-4]|[Qq]uarter\s+[1-4]|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter)(?:\s+of\s+)?\b'
        quarters = re.findall(quarter_pattern, text)
        
        # Month pattern
        month_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b'
        months = re.findall(month_pattern, text)
        
        # Add extracted periods
        periods.extend(years)
        periods.extend(quarters)
        periods.extend(months)
        
        return periods
    
    async def analyze_revenue_trends(self, documents: List[Dict[str, Any]], ticker: str) -> Dict[str, Any]:
        """
        Analyze revenue trends from financial documents
        
        Args:
            documents: List of financial documents
            ticker: Company ticker symbol
            
        Returns:
            Analysis results
        """
        try:
            # Extract revenue data from documents
            revenue_data = []
            
            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # Extract revenue values
                financials = self.extract_financial_values(content)
                
                # Process revenue values
                for revenue_item in financials["revenues"]:
                    # Skip if no time periods found
                    if not revenue_item["time_periods"]:
                        continue
                    
                    # Use the first time period
                    time_period = revenue_item["time_periods"][0]
                    
                    # Add revenue data point
                    revenue_data.append({
                        "ticker": ticker,
                        "time_period": time_period,
                        "revenue": revenue_item["value"],
                        "context": revenue_item["context"],
                        "source": doc.get('source', '')
                    })
            
            # If no revenue data found
            if not revenue_data:
                return {
                    "ticker": ticker,
                    "status": "No revenue data found",
                    "trends": []
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(revenue_data)
            
            # Sort by time period (assuming time periods are sortable)
            # This is simplistic; real implementation would need better date parsing
            try:
                df = df.sort_values("time_period")
            except:
                pass
            
            # Calculate growth rates if more than one data point
            trends = []
            if len(df) > 1:
                df['previous_revenue'] = df['revenue'].shift(1)
                df['growth_rate'] = (df['revenue'] - df['previous_revenue']) / df['previous_revenue'] * 100
                
                for _, row in df.iterrows():
                    if pd.notnull(row.get('growth_rate')):
                        trends.append({
                            "time_period": row['time_period'],
                            "revenue": row['revenue'],
                            "growth_rate": row['growth_rate'],
                            "source": row['source']
                        })
            
            return {
                "ticker": ticker,
                "status": "success",
                "revenue_data": revenue_data,
                "trends": trends
            }
            
        except Exception as e:
            logger.error(f"Error analyzing revenue trends: {str(e)}")
            return {
                "ticker": ticker,
                "status": "error",
                "message": str(e),
                "trends": []
            }
    
    async def calculate_financial_ratios(self, financial_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate financial ratios from financial statement data
        
        Args:
            financial_data: Financial statement data
            
        Returns:
            Dictionary of calculated ratios
        """
        try:
            ratios = {}
            
            # Extract data from financial statements
            income_stmt = financial_data.get("income_statement", [])
            balance_sheet = financial_data.get("balance_sheet", [])
            
            if not income_stmt or not balance_sheet:
                return ratios
            
            # Get the most recent income statement and balance sheet
            latest_income = income_stmt[0] if income_stmt else {}
            latest_balance = balance_sheet[0] if balance_sheet else {}
            
            # Calculate profitability ratios
            if 'Total Revenue' in latest_income and latest_income['Total Revenue'] > 0:
                # Gross margin
                if 'Gross Profit' in latest_income:
                    ratios['gross_margin'] = latest_income['Gross Profit'] / latest_income['Total Revenue'] * 100
                
                # Net margin
                if 'Net Income' in latest_income:
                    ratios['net_margin'] = latest_income['Net Income'] / latest_income['Total Revenue'] * 100
                
                # Operating margin
                if 'Operating Income' in latest_income:
                    ratios['operating_margin'] = latest_income['Operating Income'] / latest_income['Total Revenue'] * 100
            
            # Calculate liquidity ratios
            if 'Total Current Liabilities' in latest_balance and latest_balance['Total Current Liabilities'] > 0:
                # Current ratio
                if 'Total Current Assets' in latest_balance:
                    ratios['current_ratio'] = latest_balance['Total Current Assets'] / latest_balance['Total Current Liabilities']
                
                # Quick ratio
                if 'Total Current Assets' in latest_balance and 'Inventory' in latest_balance:
                    quick_assets = latest_balance['Total Current Assets'] - latest_balance['Inventory']
                    ratios['quick_ratio'] = quick_assets / latest_balance['Total Current Liabilities']
            
            # Calculate solvency ratios
            if 'Total Assets' in latest_balance and latest_balance['Total Assets'] > 0:
                # Debt to assets
                if 'Total Debt' in latest_balance:
                    ratios['debt_to_assets'] = latest_balance['Total Debt'] / latest_balance['Total Assets']
                
                # Return on assets (ROA)
                if 'Net Income' in latest_income:
                    ratios['roa'] = latest_income['Net Income'] / latest_balance['Total Assets'] * 100
            
            # Return on equity (ROE)
            if 'Total Stockholder Equity' in latest_balance and latest_balance['Total Stockholder Equity'] > 0:
                if 'Net Income' in latest_income:
                    ratios['roe'] = latest_income['Net Income'] / latest_balance['Total Stockholder Equity'] * 100
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {str(e)}")
            return {}