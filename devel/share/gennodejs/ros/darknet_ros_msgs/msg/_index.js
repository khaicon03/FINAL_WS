
"use strict";

let BoundingBox = require('./BoundingBox.js');
let BoundingBoxes = require('./BoundingBoxes.js');
let ObjectCount = require('./ObjectCount.js');
let CheckForObjectsActionResult = require('./CheckForObjectsActionResult.js');
let CheckForObjectsAction = require('./CheckForObjectsAction.js');
let CheckForObjectsGoal = require('./CheckForObjectsGoal.js');
let CheckForObjectsFeedback = require('./CheckForObjectsFeedback.js');
let CheckForObjectsActionFeedback = require('./CheckForObjectsActionFeedback.js');
let CheckForObjectsActionGoal = require('./CheckForObjectsActionGoal.js');
let CheckForObjectsResult = require('./CheckForObjectsResult.js');

module.exports = {
  BoundingBox: BoundingBox,
  BoundingBoxes: BoundingBoxes,
  ObjectCount: ObjectCount,
  CheckForObjectsActionResult: CheckForObjectsActionResult,
  CheckForObjectsAction: CheckForObjectsAction,
  CheckForObjectsGoal: CheckForObjectsGoal,
  CheckForObjectsFeedback: CheckForObjectsFeedback,
  CheckForObjectsActionFeedback: CheckForObjectsActionFeedback,
  CheckForObjectsActionGoal: CheckForObjectsActionGoal,
  CheckForObjectsResult: CheckForObjectsResult,
};
