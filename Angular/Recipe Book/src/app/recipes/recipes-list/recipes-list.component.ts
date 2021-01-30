import { Component, OnInit } from '@angular/core';
import { Recipe } from '../recipe.module';

@Component({
  selector: 'app-recipes-list',
  templateUrl: './recipes-list.component.html',
  styleUrls: ['./recipes-list.component.css']
})
export class RecipesListComponent implements OnInit {

  recipes: Recipe[] = [new Recipe('Recipe name test 1', 'Recipe desc test 1', 'https://d3o5sihylz93ps.cloudfront.net/wp-content/uploads/2019/11/11165704/shutterstock_1479089261.jpg')
  , new Recipe('Recipe name test 2', 'Recipe desc test 2', 'https://d3o5sihylz93ps.cloudfront.net/wp-content/uploads/2019/11/11165704/shutterstock_1479089261.jpg')];

  constructor() { }

  ngOnInit(): void {
  }

}
